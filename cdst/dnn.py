import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold, train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import ray
from ray import tune

import os
import itertools
import warnings
import filelock

import math
import string
import time
import random

from .shared import *

############################ Deep Learning Architecture ############################

class dnn(nn.Module):

    # Constructor
    def __init__(self, in_feat, layers, dropout_p=None, act_fn=torch.relu):
        super(dnn, self).__init__()
        layers = [in_feat] + layers   # Add input layer
        self.hidden = nn.ModuleList()
        self.out = nn.Linear(layers[-1],1).double()
        self.act_fn = act_fn
        self.dropout = nn.Dropout(p=dropout_p)
        # --- Scalable Layers ---
        for input_size, output_size in zip(layers, layers[1:]):
            self.hidden.append(nn.Linear(input_size,output_size).double())
            
    # Prediction
    def forward(self, x):
        L = len(self.hidden)
        for (l, single_layer) in zip(range(L), self.hidden):
            x = single_layer(x)
            x = self.dropout(self.act_fn(x))
        x = self.act_fn(self.out(x))
        return x
    
    
############################ Training Procedure ############################

# Training procedure
def train_dnn_raytune(config, 
                      num_in_feat,
                      criterion=nn.MSELoss(),
                      checkpoint_dir=None, 
                      num_epochs=100, 
                      train_dataset=None, 
                      valid_dataset=None,
                      metric_dict={'rmse':     lambda y_est,y: torch.sqrt(nn.MSELoss(reduction="mean")(y_est,y)),
                                   'mean_l1':  lambda y_est,y: nn.L1Loss(reduction="mean")(y_est,y),
                                   'l1_iqr':   lambda y_est,y: compute_iqr(nn.L1Loss(reduction="none")(y_est,y)),
                                   'med-ape':  lambda y_est,y: torch.median((y-y_est).abs()/y.abs()),
                                   'mape':     lambda y_est,y: torch.mean((y-y_est).abs()/y.abs()),
                                   'mape_iqr': lambda y_est,y: compute_iqr((y-y_est).abs()/y.abs())},
                      train_metric_samples=None,
                      force_cpu=False
                     ):
    '''
    Training procedure for cspd regression with Ray Tune hyperparameter tuning
    This function is to be used for training with hyperparameter tuning based on Ray Tune. A cspd architecture table is given and the following hyperparameters are sampled by Ray Tune:
        lr: learning rate
        h_branch: neural network architecture definition
        dropout_p: dropout probability of all the neurons in the network
        k: k-fold index k for the dataset
        batch_size: the batch size use for the mini-batch use for batch gradient descent

    Args:
    (Note: This function is not meant to run directly by user, these arguemnts are passed indirectly by tune.run.)
        config: Ray Tune hyperparameter sampling configuration (for details, please refer to: https://docs.ray.io/en/master/tune/user-guide.html)
        checkpoint_dir: Output directory of training log, including the tensorboard output
        num_epochs: Number of training epochs
        num_in_feat: Number of input features for the network
        num_branch: Number of parallel branches in the network (segments)
        train_dataset: (List of or single) OutputDataBatch class Pytorch dataloader object
        valid_dataset: (List of or single) OutputDataBatch class Pytorch dataloader object
        metric_dict: Dictionary of loss function to be use for metric reporting (Attention: These are only used for reporting, not as training loss function!)

    Returns:
        result is return indirectly with tune.run

    Raises:
        -

    Example:
        -
    '''
    #====================== Ray Tune Parameters Setup ======================#

    if 'dropout_p' in config.keys():
        _dropout_p = config['dropout_p']
    else:
        _dropout_p = 0

    if 'batch_size' in config.keys():
        _batch_size = config['batch_size']
    else:
        _batch_size = 1
        
    # determine if input is k-fold dataset or single dataset
    if (type(train_dataset) is list) and (type(valid_dataset) is list) and ('k' in config.keys()):
        _train_dataset = train_dataset[config['k']]
        _valid_dataset = valid_dataset[config['k']]
    else:
        _train_dataset = train_dataset
        _valid_dataset = valid_dataset
        
    # measure error metric across whole epoch if no sample length is given
    # (the latest progress might not be shown properly and error could be overestimated by earlier samples)
    if train_metric_samples is None:
        train_metric_samples = len(_train_dataset)
    
    # gpu usage
    if not force_cpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"

    train_loader = torch.utils.data.DataLoader(dataset=_train_dataset, batch_size=_batch_size, shuffle=True, 
                                               collate_fn=lambda x: [ x_ele.to(device) for x_ele in default_collate(x) ] )
    valid_loader = torch.utils.data.DataLoader(dataset=_valid_dataset, batch_size=_batch_size, shuffle=True,
                                               collate_fn=lambda x: [ x_ele.to(device) for x_ele in default_collate(x) ] )

    #====================== Model Setup ======================#

    # initialize ANN architecture
    model = dnn(in_feat = num_in_feat, 
                layers = config['h_layers'], 
                dropout_p = _dropout_p,
                act_fn = torch.relu)
    model.apply(initialize_weights)

    # gpu usage
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)   # for multiple GPUs
    model.to(device)

    
    # optimizer is controlled by ray tune hyperparameter
    optimizer = torch.optim.Adam(model.parameters(), lr = config["lr"])
    
    # The `checkpoint_dir` parameter gets passed by Ray Tune when a checkpoint
    # should be restored.
    if checkpoint_dir:
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        model_state, optimizer_state = torch.load(checkpoint)
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
    
    # create loss metric dictionary to store results
    history = {'train': {}, 'valid': {}}
    metric_output = {}

    for epoch in range(num_epochs):

        #====================== Training ======================#

        # training using all training samples
        for i, (x, y) in enumerate(train_loader):
            # zero the parameter gradients
            optimizer.zero_grad()

            # set the model to training mode
            model.train()
            
            # forward + backward + optimize
            y_est = model(x)
            loss = criterion(y_est, y)
            loss.backward()
            optimizer.step()
            
            # record the prediction results
            # :: the following function is use to remove zero-padded samples in batch training
            # :: loss metrics are kept in a FIFO queue per latest samples in order to compute statistics
            history['train'] = loss_fifo(y_est, y, history=history['train'], queue_len=train_metric_samples)
            
            
        #====================== Validation ======================#
        
        # set the model to evaluation mode
        model.eval()

        # training using all validation samples
        with torch.no_grad():
            for  i, (x, y) in enumerate(valid_loader):
                y_est = model(x)
                # record the prediction results
                # :: the following function is use to remove zero-padded samples in batch training
                # :: loss metrics are kept in a FIFO queue per latest samples in order to compute statistics
                history['valid'] = loss_fifo(y_est, y, history=history['valid'], queue_len=len(_valid_dataset))
                
        for metric in metric_dict.keys():
            for dataset in history.keys():
                metric_label = '_'.join([dataset,metric])
                metric_output[metric_label] = metric_dict[metric](history[dataset]['y_est'],history[dataset]['y']).item()
                
        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and will potentially be passed as the `checkpoint_dir`
        # parameter in future iterations.
        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save( (model.state_dict(), optimizer.state_dict()), path )

        tune.report(**metric_output)
        

# Training procedure
def train_dnn(model, train_dataset, valid_dataset, criterion, optimizer, 
              epochs=100, 
              batch_size=1, 
              metric_dict={'rmse':     lambda y_est,y: torch.sqrt(nn.MSELoss(reduction="mean")(y_est,y)),
                           'mean_l1':  lambda y_est,y: nn.L1Loss(reduction="mean")(y_est,y),
                           'l1_iqr':   lambda y_est,y: compute_iqr(nn.L1Loss(reduction="none")(y_est,y)),
                           'med-ape':  lambda y_est,y: torch.median((y-y_est).abs()/y.abs()),
                           'mape':     lambda y_est,y: torch.mean((y-y_est).abs()/y.abs()),
                           'mape_iqr': lambda y_est,y: compute_iqr((y-y_est).abs()/y.abs())},
              train_metric_samples=None,
              ):
    '''
    Training procedure for cspd regression
    This function is to be used for training without hyperparameter optimization, this function is usually use for test run to make sure all modification on the cspd architecture is working before submitting a list of models for hyperparameter search. To use hyperparameter optimization, please use either `train_cspd_raytune` or `train_cspd_raytune_auto_architecture`.
    
    Args:
        model: Pytorch model object of cspd
        train_dataset: OutputDataBatch or OutputData class Pytorch dataloader object
        valid_dataset: OutputDataBatch or OutputData class Pytorch dataloader object
        criterion: Training criterion to be used (eg. criterion = nn.MSELoss())
        optimizer: Training optimizer to be used (eg. optimizer = torch.optim.Adam(model.parameters(), lr = 0.1))
        epochs: Number of training epochs to be used
        batch_size: The batch size to use for batch gradient descent of the output dimension, the input dimension will be setted to zero patching within the OutputDataBatch object for comparable input size to perform the stacked computation
        metric_dict: Dictionary of loss function to be use for metric reporting (Attention: These are only used for reporting, not as training loss function!)
        history_queue_len: The number of loss result samples to keep for statistical reporting

    Returns:
        history: Training and validation results summary
        model: Implicitly updated in the model object

    Raises:
        -

    Example:
        # Example of cspd training with no subgroupings
        # (Remark: s_model and s_test are all generated with all 1's by model_test_split function with ohe_cols=None)
        (x_model, x_test, s_model, s_test, y_model, y_test) = model_test_split(x, y, ohe_cols=None, id_col=y_id_col, test_size=0.3, random_state=25)
        dataset_train = OutputDataBatch(x_model, s_model, y_model, zero_patch = False)
        dataset_valid = OutputDataBatch(x_test, s_test, y_test, zero_patch = False)
        architecture = [2,2,3,2,2]   # single branch with 5 layers
        model = dnn(in_feat=10, layers=architecture, dropout_p=0.3)
        model.apply(initialize_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.1)
        criterion = nn.MSELoss()
        metric_dict = {'rmse': lambda y_est,y: torch.sqrt(nn.MSELoss(reduction="none")(y_est,y)), 
                       'mape': lambda y_est,y: (y-y_est).abs()/y.abs()}
        training_results = train_cspd(model=model, 
                                      train_dataset=dataset_model, 
                                      valid_dataset=dataset_test, 
                                      criterion=criterion,
                                      optimizer=optimizer,
                                      metric_dict=metric_dict,
                                      epochs=num_epochs, 
                                      batch_size=64)
    Author:
        Dr. Calvin Chan
        calvin.chan@bayer.com
    '''
    history = {'train': {}, 'valid': {}}
    metric_output = {}

    if train_metric_samples is None:
        train_metric_samples = len(train_dataset)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):

        #====================== Training ======================#
        running_loss = 0.0
        epoch_steps = 0

        # training using all training samples
        for i, (x, y) in enumerate(train_loader):
            # zero the parameter gradients
            optimizer.zero_grad()

            # set the model to training mode
            model.train()
            
            # forward + backward + optimize
            y_est = model(x)
            loss = criterion(y_est, y)
            loss.backward()
            optimizer.step()
            history['train'] = loss_fifo(y_est, y, history=history['train'], queue_len=train_metric_samples)

        #====================== Validation ======================#
        
        # set the model to evaluation mode
        model.eval()

        # training using all validation samples
        with torch.no_grad():
            for  i, (x, y) in enumerate(valid_loader):
                y_est = model(x)
                history['valid'] = loss_fifo(y_est, y, history=history['valid'], queue_len=len(valid_dataset))
    
        metric_labels = []
        for metric in metric_dict.keys():
            for dataset in history.keys():
                metric_label = '_'.join([dataset,metric])
                metric_output[metric_label] = metric_dict[metric](history[dataset]['y_est'],history[dataset]['y']).item()
                metric_labels.append(metric_label)
                        
        print(f"[Epoch: { epoch+1 }]", end=" " )
        for metric_label in metric_labels:
            print(f"{metric_label}: {metric_output[metric_label]:.3f},", end=" ")
        print(f"")

    return (history)


def train_dnn_raytune_cpu_gpu_distributed(config, 
                                          num_in_feat,
                                          criterion=nn.MSELoss(),
                                          checkpoint_dir=None, 
                                          num_epochs=100, 
                                          train_dataset=None, 
                                          valid_dataset=None,
                                          metric_dict={'rmse':     lambda y_est,y: torch.sqrt(nn.MSELoss(reduction="mean")(y_est,y)),
                                                       'mean_l1':  lambda y_est,y: nn.L1Loss(reduction="mean")(y_est,y),
                                                       'l1_iqr':   lambda y_est,y: compute_iqr(nn.L1Loss(reduction="none")(y_est,y)),
                                                       'med-ape':  lambda y_est,y: torch.median((y-y_est).abs()/y.abs()),
                                                       'mape':     lambda y_est,y: torch.mean((y-y_est).abs()/y.abs()),
                                                       'mape_iqr': lambda y_est,y: compute_iqr((y-y_est).abs()/y.abs())},
                                          train_metric_samples=None,
                                          ):
    '''
    CPU/GPU Distributed Wrapper Function for Training procedure for cspd regression
    This function is written to allow training done on both CPU and GPU of a single machine at the same time.
    
    Args:

    Returns:
        result: Training metric results

    Source:
        This code is modified from the following: https://discuss.ray.io/t/different-trial-on-cpu-and-gpu-separately/2883

    Author:
        Dr. Calvin Chan
        calvin.chan@bayer.com
    '''
    
    lock_filepath = "/tmp/gpu.lock"
    a = filelock.FileLock(lock_filepath)
#     os.chmod(lock_filepath, 777)
    try:
        # Makes it so that 1 trial will use the GPU at once.
        a.acquire(timeout=1)
        result = train_dnn_raytune(config, 
                                   num_in_feat,
                                   criterion,
                                   checkpoint_dir, 
                                   num_epochs, 
                                   train_dataset, 
                                   valid_dataset,
                                   metric_dict,
                                   train_metric_samples,
                                   force_cpu=False
                                   )
    except filelock.Timeout:
        # If the lock is acquired, you can just use CPU, and disable GPU access.
        result = train_dnn_raytune(config, 
                                   num_in_feat,
                                   criterion,
                                   checkpoint_dir, 
                                   num_epochs, 
                                   train_dataset, 
                                   valid_dataset,
                                   metric_dict,
                                   train_metric_samples,
                                   force_cpu=True
                                   )
    finally:
        # Release the lock after training is done.
        a.release()
    return result
