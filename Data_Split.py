import os
import csv
import logging
import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from sklearn.metrics import auc, mean_squared_error, precision_recall_curve, roc_auc_score, r2_score
from fpgnn.data import MoleDataSet, MoleData, scaffold_split
from fpgnn.model import FPGNN

def mkdir(path, isdir=True):
    if not isdir:
        path = os.path.dirname(path)
    if path != '':
        os.makedirs(path, exist_ok=True)

def set_log(name, save_path):
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    
    log_stream = logging.StreamHandler()
    log_stream.setLevel(logging.DEBUG)
    log.addHandler(log_stream)
    
    mkdir(save_path)
    
    log_file_d = logging.FileHandler(os.path.join(save_path, 'debug.log'))
    log_file_d.setLevel(logging.DEBUG)
    log.addHandler(log_file_d)
    
    return log

def get_header(path):
    with open(path) as file:
        header = next(csv.reader(file))
    return header

def get_task_name(path):
    task_name = get_header(path)[1:]
    return task_name

def load_data(path, args):
    with open(path) as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        lines = []
        for line in reader:
            lines.append(line)
        data = []
        for line in lines:
            one = MoleData(line, args)
            data.append(one)
        data = MoleDataSet(data)
        
        fir_data_len = len(data)
        data_val = []
        smi_exist = []
        for i in range(fir_data_len):
            if data[i].mol is not None:
                smi_exist.append(i)
        data_val = MoleDataSet([data[i] for i in smi_exist])
        now_data_len = len(data_val)
        print('There are', now_data_len, 'smiles in total.')
        if fir_data_len - now_data_len > 0:
            print('There are', fir_data_len, 'smiles first, but', fir_data_len - now_data_len, 'smiles is invalid.')
        
    return data_val

def split_data(data, type, size, seed, log):
    assert len(size) == 3
    assert sum(size) == 1
    
    if type == 'random':
        data.random_data(seed)
        train_size = int(size[0] * len(data))
        val_size = int(size[1] * len(data))
        train_val_size = train_size + val_size
        train_data = data[:train_size]
        val_data = data[train_size:train_val_size]
        test_data = data[train_val_size:]
    
    elif type == 'scaffold':
        train_data, val_data, test_data = scaffold_split(data, size, seed, log)
    else:
        raise ValueError('Split_type is Error.')
    
    # Save the split datasets to CSV files
    save_data_to_csv(train_data, 'train_data.csv')
    save_data_to_csv(val_data, 'val_data.csv')
    save_data_to_csv(test_data, 'test_data.csv')
    
    return MoleDataSet(train_data), MoleDataSet(val_data), MoleDataSet(test_data)

def save_data_to_csv(data, file_name):
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        for row in data.to_iterable():  # Assuming data has a method to_iterable
            writer.writerow(row)