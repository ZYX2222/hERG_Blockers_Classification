# hERG_Blokers_Classification  
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

## Mains  
data: hERG blockade dataset (csv file contains SMILES and class label) for developing models, which can be replaced by your own dataset;  

fpgnn: main components of the model.  

## Commands  

### train  
Use train.py  

Args:  

data_path : The path of input CSV file. E.g. input.csv  
dataset_type : The type of dataset. E.g. classification or regression  
save_path : The path to save output model. E.g. model_save  
log_path : The path to record and save the result of training. E.g. log  
E.g.  

python train.py  --data_path data/test.csv  --dataset_type classification  --save_path model_save  --log_path log  

### Hyperparameters Optimization  




