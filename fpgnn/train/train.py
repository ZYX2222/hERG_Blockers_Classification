from argparse import Namespace
from logging import Logger
import os
import csv
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from fpgnn.tool.tool import mkdir, get_task_name, load_data, split_data, get_label_scaler, get_loss, get_metric, save_model, NoamLR, load_model
from fpgnn.model import FPGNN
from fpgnn.data import MoleDataSet

def epoch_train(model,data,loss_f,optimizer,scheduler,args):
    model.train()
    data.random_data(args.seed)
    loss_sum = 0
    data_used = 0
    iter_step = args.batch_size
    
    for i in range(0,len(data),iter_step):
        if data_used + iter_step > len(data):
            break
        
        data_now = MoleDataSet(data[i:i+iter_step])
        smile = data_now.smile()
        label = data_now.label()

        mask = torch.Tensor([[x is not None for x in tb] for tb in label])
        target = torch.Tensor([[0 if x is None else x for x in tb] for tb in label])
        
        if next(model.parameters()).is_cuda:
            mask, target = mask.cuda(), target.cuda()
        
        weight = torch.ones(target.shape)
        if args.cuda:
            weight = weight.cuda()
        
        model.zero_grad()
        pred = model(smile)
        loss = loss_f(pred,target) * weight * mask
        loss = loss.sum() / mask.sum()
        loss_sum += loss.item()
        data_used += len(smile)
        loss.backward()
        optimizer.step()
        if isinstance(scheduler, NoamLR):
            scheduler.step()
    if isinstance(scheduler, ExponentialLR):
        scheduler.step()

def predict(model,data,batch_size,scaler):
    model.eval()
    pred = []
    data_total = len(data)
    
    for i in range(0,data_total,batch_size):
        data_now = MoleDataSet(data[i:i+batch_size])
        smile = data_now.smile()
        
        with torch.no_grad():
            pred_now = model(smile)
        
        pred_now = pred_now.data.cpu().numpy()
        
        if scaler is not None:
            ave = scaler[0]
            std = scaler[1]
            pred_now = np.array(pred_now).astype(float)
            change_1 = pred_now * std + ave
            pred_now = np.where(np.isnan(change_1),None,change_1)
        
        pred_now = pred_now.tolist()
        pred.extend(pred_now)
    
    return pred


def compute_score(pred, label, metrics_dict, args, log):
    info = log.info

    if len(pred) == 0:
        return {metric: [float('nan')] * args.task_num for metric in metrics_dict}

    results = {metric: [] for metric in metrics_dict}
    for metric, func in metrics_dict.items():
        for i in range(args.task_num):
            pred_val_i = [pred[j][i] for j in range(len(pred)) if label[j][i] is not None]
            label_val_i = [label[j][i] for j in range(len(pred)) if label[j][i] is not None]

            if args.dataset_type == 'classification':
                if all(one == 0 for one in label_val_i) or all(one == 1 for one in label_val_i):
                    info(f'Warning: All labels are 1 or 0 for metric {metric}.')
                    results[metric].append(float('nan'))
                    continue
                if all(one == 0 for one in pred_val_i) or all(one == 1 for one in pred_val_i):
                    info(f'Warning: All predictions are 1 or 0 for metric {metric}.')
                    results[metric].append(float('nan'))
                    continue
            results[metric].append(func(label_val_i, pred_val_i))

    return results

def fold_train(args, log):
    info = log.info
    debug = log.debug
    
    debug('Start loading data')
    
    args.task_names = get_task_name(args.data_path)
    data = load_data(args.data_path,args)
    args.task_num = data.task_num()
    data_type = args.dataset_type
    if args.task_num > 1:
        args.is_multitask = 1
    
    debug(f'Splitting dataset with Seed = {args.seed}.')
    if args.val_path:
        val_data = load_data(args.val_path,args)
    if args.test_path:
        test_data = load_data(args.test_path,args)
    if args.val_path and args.test_path:
        train_data = data
    elif args.val_path:
        split_ratio = (args.split_ratio[0],0,args.split_ratio[2])
        train_data, _, test_data = split_data(data,args.split_type,split_ratio,args.seed,log)
    elif args.test_path:
        split_ratio = (args.split_ratio[0],args.split_ratio[1],0)
        train_data, val_data, _ = split_data(data,args.split_type,split_ratio,args.seed,log)
    else:
        train_data, val_data, test_data = split_data(data,args.split_type,args.split_ratio,args.seed,log)
    debug(f'Dataset size: {len(data)}    Train size: {len(train_data)}    Val size: {len(val_data)}    Test size: {len(test_data)}')
    
    if data_type == 'regression':
        label_scaler = get_label_scaler(train_data)
    else:
        label_scaler = None
    args.train_data_size = len(train_data)
    
    loss_f = get_loss(data_type)
    metrics_dict = get_metric(args.metric)
    
    debug('Training Model')
    model = FPGNN(args)
    debug(model)
    if args.cuda:
        model = model.to(torch.device("cuda"))
    save_model(os.path.join(args.save_path, 'model.pt'),model,label_scaler,args)
    optimizer = Adam(params=model.parameters(), lr=args.init_lr, weight_decay=0)
    scheduler = NoamLR( optimizer = optimizer,  warmup_epochs = [args.warmup_epochs], total_epochs = None or [args.epochs] * args.num_lrs, \
                        steps_per_epoch = args.train_data_size // args.batch_size, init_lr = [args.init_lr], max_lr = [args.max_lr], \
                        final_lr = [args.final_lr] )
    if data_type == 'classification':
        best_score = -float('inf')
    else:
        best_score = float('inf')
    best_epoch = 0
    n_iter = 0
    
    for epoch in range(args.epochs):
        info(f'Epoch {epoch}')
        
        epoch_train(model, train_data, loss_f, optimizer, scheduler, args)
        
        train_pred = predict(model, train_data, args.batch_size, label_scaler)
        train_label = train_data.label()
        train_scores = compute_score(train_pred, train_label, metrics_dict, args, log)
        val_pred = predict(model, val_data, args.batch_size, label_scaler)
        val_label = val_data.label()
        val_scores = compute_score(val_pred, val_label, metrics_dict, args, log)
        
        # Output train scores
        for metric, scores in train_scores.items():
            ave_score = np.nanmean(scores)
            info(f'Train {metric} = {ave_score:.6f}')

        for metric, scores in val_scores.items():
            ave_score = np.nanmean(scores)
            info(f'Validation {metric} = {ave_score:.6f}')
        
        best_metric = 'aroc' if data_type == 'classification' else 'r2'
        ave_val_score = np.nanmean(val_scores[best_metric]) 


        if (data_type == 'classification' and ave_val_score > best_score) or \
           (data_type == 'regression' and ave_val_score > best_score):
            best_score = ave_val_score
            best_epoch = epoch
            save_model(os.path.join(args.save_path, 'model.pt'), model, label_scaler, args)

    info(f'Best validation {best_metric} = {best_score:.6f} on epoch {best_epoch}')
    
    model = load_model(os.path.join(args.save_path, 'model.pt'), args.cuda, log)
    test_smile = test_data.smile()
    test_label = test_data.label()

    test_pred = predict(model, test_data, args.batch_size, label_scaler)
    test_scores = compute_score(test_pred, test_label, metrics_dict, args, log)

    for metric, scores in test_scores.items():
        if isinstance(scores, list) and len(scores) > 0:
            ave_test_score = np.nanmean(scores) if np.array(scores).ndim == 1 else np.nanmean(scores, axis=1)
            info(f'{metric} test score: {ave_test_score:.6f}')
            if args.task_num > 1:
                for task_name, score in zip(args.task_names, scores):
                    info(f'Task {task_name} {metric} = {score:.6f}')
    
    return test_scores 