# hERG_Blockers_Classification  
## Environment

The most important python packages are:  

python == 3.11.8    
pytorch == 2.2.1    
pytorch-cuda == 11.8    
tensorboard == 2.16.2    
rdkit == 2023.9.5  
scikit-learn == 1.4.1.post1    
hyperopt == 0.2.7    
numpy == 1.26.4  

To replicate or devleop models more conveniently, the environment file <environment.txt> is provided to install environment directly. 

command:   
```
conda create --name new_env --file environment.txt
```


## Mains  
### data  
The dataset (csv file contains SMILES and class label) for developing models, which can be replaced by your own dataset;  

Note: For training, the csv file should includes SMILES and label; for predicting, the csv file should includes SMILES.   

### fpgnn  
Main components of the model.  

### saved_model  
The best trained model (model.pt).

## Commands  

### train  
Use train.py  

Args:  

data_path : The path of input CSV file. E.g. input.csv  
dataset_type : The type of dataset. E.g. classification or regression  
save_path : The path to save output model. E.g. model_save  
log_path : The path to record and save the result of training. E.g. log 

E.g.  

python train.py  --data_path data/test.csv  --dataset_type classification  --save_path saved_model  --log_path log  

### Hyperparameters Optimization  
Use hyper_opti.py  

Args:  

data_path : The path of input CSV file. E.g. input.csv  
dataset_type : The type of dataset. E.g. classification or regression  
save_path : The path to save output model. E.g. saved_model  
log_path : The path to record and save the result of hyperparameters optimization. E.g. log  

E.g.  

python hyper_opti.py  --data_path data/test.csv  --dataset_type classification  --save_path saved_model  --log_path log  

### Predict  
Use predict.py  

Args:  

predict_path : The path of input CSV file to predict. E.g. input.csv  
result_path : The path of output CSV file. E.g. output.csv  
model_path : The path of trained model. E.g. saved_model/model.pt  

E.g.  

python predict.py  --predict_path data/test.csv  --model_path saved_model/model.pt  --result_path result.csv  


## Acknowledgments  
The code origninated from previous reference: 10.1093/bib/bbac408, with the addition or replacement of counted fingerprint inputs in the current study.




