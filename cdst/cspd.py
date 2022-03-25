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

class SAMPLE_Branch(nn.Module):

    # Constructor
    def __init__(self, in_feat, layers, dropout_p=None, act_fn=torch.relu):
        super(SAMPLE_Branch, self).__init__()
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
            x = torch.unbind(x,dim=1)                                        # Separate activation sum
            x = [ self.dropout(self.act_fn(x_sample)) for x_sample in x ]    # Compute sum activation for each X_hcp separately
            x = torch.stack(x, dim=1)                                        # Stack hcp_samples back together for next layer
        x = torch.unbind(x,dim=1)
        x = [ self.act_fn(x_sample.sum(dim=1)) for x_sample in x ]   # Compute sum activation for each X_hcp separately
        x = torch.stack(x,dim=1)                                     # Stack back together for output

        return x
    
class cspd(nn.Module):
    
    # Constructor
    def __init__(self, in_feat, Subgroups, Architecture, dropout_p=None, act_fn=torch.relu):
        super(cspd, self).__init__()
        self.hidden = nn.ModuleList()
        # --- Scalable Subgroup Branch ---
        for s_id in range(Subgroups):
            self.hidden.append(SAMPLE_Branch(in_feat, Architecture[s_id], dropout_p, act_fn))
            
    # Prediction
    def forward(self, x, s):
        z = [ z_s(x) for z_s in self.hidden ]
        z = torch.stack(z, dim=2)
        branch_avg_factor = s.sum(dim=[2], keepdim=True)             # Cases with unsure subgrouping, normalize them across all subgroups
        branch_avg_factor = torch.max(branch_avg_factor,             # Avoid divide by zero
                                      torch.ones(branch_avg_factor.shape, 
                                                 device=branch_avg_factor.device.type))   
        z = (z * s).sum(dim=[2], keepdim=True) / branch_avg_factor   # Sum across subgroup branches
        z = z.sum(dim=[1], keepdim=True)                             # Sum across samples
        return z
        
        
############################ Data Handling ############################
    
def sgrp_split_parser(x, y, test_ratio=0.2, sgrp_col=None, y_id_col=None, report_id=False, random_state=25):
    '''
    Parsing Subgroup and Dualscaled Data
    Split the dualscaled x and y dataframe into a list of dataframe and extract the subgrouping information for Pytorch data object
    
    Args:
        x: Pandas dataframe at same or higher resoultion than y
        y: Pandas dataframe at same or lower resolution than x
        test_ratio: The split ratio of the test set
        sgrp_col: Column name of the column(s) indicating the branch/subgroup/segregation in x
        y_id_col: Common column for dualscaled x and y, the column id name which joins the 2 dataframes

    Returns:
        out: Tuple of 6 variables - x_model, x_test, s_model, s_test, y_model, y_test indicates the splited dataset
        s_ohe_table: Subgroup one-hot-encoding conversion table
            
    Author:
        Dr. Calvin Chan
        calvin.chan@bayer.com
    '''
    if sgrp_col is None:
        s_ohe = pd.DataFrame(np.ones([x.shape[0],1]), columns=['no_subgroup'])
        s_ohe.index = x.index
        s_ohe_table = None
    else:
        (s_ohe, s_ohe_table) = combine_multidimensional_ohe(x[[sgrp_col]])
        x = x.drop(sgrp_col,axis=1)
        
    if y_id_col is not None:
        s_ohe[y_id_col] = x[y_id_col]
        
    out, sample_ids = model_test_split(x, s_ohe, y, id_col=y_id_col, test_ratio=test_ratio, random_state=random_state, report_id=True)
    
    if report_id:
        return (out, s_ohe_table, sample_ids)
    else:
        return (out, s_ohe_table)
        
        
############################ Pytorch Data Object ############################

class SgrpDataBatch(Dataset):
    '''
    Data object for precomputing zero-patched dataset
    
    This is the data class for pytorch dataloader for input dataset with subgrouping/segregation 
    input designed specifically for input x with higher resolution than output y.  Due to the
    difference between the resolution between each input sample, zero patching is required to
    perform mini-batch training. 
    
    This class precompute the zero-patched samples and provides an internal flag to decide whether
    a zero-patched sample is outputted during loading.

    The initializing input data x, s, y are all converted to list of dataframe already by the 
    data spliting function `sgrp_split_parser`.  This is neede because the x and y have different
    resolution, in order to match the resolution, splitting must be done prior creating the dataset
    object.

    Author:
        Dr. Calvin Chan
        calvin.chan@bayer.com
    '''

    def __init__(self, x, s, y, transform=None, zero_patch=False, dtype=torch.double, samples_id=None):
        
        input_equal_length_flag = (len(set([len(y), len(x), len(s)])) == 1)
        if input_equal_length_flag:
            print("Number of x and y samples do not match!")
#         assert input_equal_length_flag, "Number of x and y samples do not match!"
        
        self.x_col = x[0].columns
        self.s_col = s[0].columns
        self.len = len(y)
        self.transform = transform
        self.zero_patched = zero_patch
        self.samples_id = samples_id
        
        # === Convert sample data type ===
        # (Common samples are extracted, therefore row ID of x,s,y must be aligned)
        self.x = self._sample_type_convert(x,dtype)
        self.s = self._sample_type_convert(s,dtype)
        self.y = self._sample_type_convert(y,dtype)
        
        # === Zero Patching x and s for Batch Training ===
        max_sample = max([ x.size(0) for x in self.x ])
        self.x_zero_patched = [ torch.cat([x.to(device), torch.zeros(max_sample - x.size(0), len(self.x_col)).to(device)], dim=0) for x in self.x ]
        self.s_zero_patched = [ torch.cat([s.to(device), torch.zeros(max_sample - s.size(0), len(self.s_col)).to(device)], dim=0) for s in self.s ]
        
    def __getitem__(self, index):
        
        if self.zero_patched:
            sample = [self.x_zero_patched[index],
                      self.s_zero_patched[index],
                      self.y[index]]
            
        else:
            sample = [self.x[index],
                      self.s[index],
                      self.y[index]]

        if self.transform:
            sample = self.transform(sample)
        return sample
    
    def __len__(self):
        return self.len
    
    def _sample_type_convert(self, samples, dtype):
        samples_out = [ torch.tensor(sample_ele.values).type(dtype) for sample_ele in samples ]
        return samples_out


############################ Data Partitioning ############################

def patitioned_data_object_cspd(x, y, sgrp_col, y_id_col, test_split_ratio, k, random_state=25, kfold_ind=False):
    # Model/Test Splitting
    ((x_model, x_test, 
      s_model, s_test, 
      y_model, y_test), 
      s_ohe_table, 
     (samples_id_model, 
      samples_id_test)) = sgrp_split_parser(x, y, 
                                           test_ratio=test_split_ratio, 
                                           sgrp_col=subgroup_col, 
                                           y_id_col=y_id_col,
                                           report_id=True,
                                           random_state=random_state)
    # K-Fold Index Sampling
    [kfold_train_ind, kfold_valid_ind] = get_k_fold_indices(n_samples=len(y_model), k=k, shuffle=False)   # Shuffle is NOT needed, since the samples were shuffled in the model/test split

    # Create K-set of datasets for Pytorch data loader
    dataset_train_kfold = [ SgrpDataBatch(select_ind(x_model,fold_ind),
                                         select_ind(s_model,fold_ind),
                                         select_ind(y_model,fold_ind),
                                         zero_patch = False,
                                         samples_id = select_ind(samples_id_model, fold_ind) )
                                               for fold_ind in kfold_train_ind ]
    dataset_valid_kfold = [ SgrpDataBatch(select_ind(x_model,fold_ind),
                                         select_ind(s_model,fold_ind),
                                         select_ind(y_model,fold_ind),
                                         zero_patch = False,
                                         samples_id = select_ind(samples_id_model, fold_ind) )
                                               for fold_ind in kfold_valid_ind ]

    dataset_model = SgrpDataBatch(x_model, s_model, y_model, zero_patch = False, samples_id = samples_id_model)
    dataset_test = SgrpDataBatch(x_test, s_test, y_test, zero_patch = False, samples_id = samples_id_test)
    
    if kfold_ind:
        return dataset_model, dataset_test, dataset_train_kfold, dataset_valid_kfold, kfold_train_ind, kfold_valid_ind
    else:
        return dataset_model, dataset_test, dataset_train_kfold, dataset_valid_kfold


############################ Training Procedure ############################

def remove_zero_padded(y_est, y, sgrp, return_length=False):
    '''
    Remove Zero Padding Data for Batch Data
    Due to the variable input resolution, zero padding is required for batch gradient decent for cspd algorithm.  Therefore, the zero padded batches could introduce a bias in the loss metric computation.  To avoid this problem, the zero padded data with all zeros for the subgroup indicator is used to remove these entries during error computation.
    
    Args:
        y_est: Model prediction output
        y: Training data output ground truth
        sgrp: Subgrouping one-hot-encoded matrix for the batch data (B x O x S matrix, where B is batch size, O is output dimensions, S is number of subgroups)

    Returns:
        out: Loss metric variable

    Raises:
        -

    Example:
        

    Author:
        Dr. Calvin Chan
        calvin.chan@bayer.com
    '''
    # identify non-zero padded data (not all subgroup flags are zero)
    # :: first `torch.any` perform non-zero padded sample checking across subgroups
    # :: second `torch.any` perform non-zero padded sample checking across input sample space within output sample
    flag = torch.any(torch.any(sgrp,dim=2,keepdim=False),dim=1,keepdim=False)   
    # select non-zero padded entries
    y_est = y_est[flag]
    y = y[flag]
    
    if not return_length:
        return(y_est, y)
    else:
        return(y_est, y, sum(flag).item())
    

# Training procedure
def train_cspd_raytune(config, 
                       num_in_feat,
                       num_branch,
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
        num_branch: Number of parallel branches in the network (subgroups)
        train_dataset: List of K element SgrpDataBatch class Pytorch dataloader object
        valid_dataset: List of K element SgrpDataBatch class Pytorch dataloader object
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

    # zero patch high-res dimension dataset if we are doing mini-batch
    if _batch_size > 1:
        _train_dataset.zero_patched = True
        _valid_dataset.zero_patched = True
    else:
        _train_dataset.zero_patched = False
        _valid_dataset.zero_patched = False

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
    model = cspd(in_feat = num_in_feat, 
                 Subgroups = num_branch, 
                 Architecture = config['h_branch'], 
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
        for i, (x, s, y) in enumerate(train_loader):
            # zero the parameter gradients
            optimizer.zero_grad()

            # set the model to training mode
            model.train()
            
            # forward + backward + optimize
            y_est = model(x, s)
            y_est, y = remove_zero_padded(y_est, y, s)
            loss = criterion(y_est, y)
            loss.backward()
            optimizer.step()
            
            # record the prediction results
            # :: the following function is use to remove zero-padded samples in batch training
            # :: loss metrics are kept in a FIFO queue per latest samples in order to compute statistics
            history['train'] = loss_fifo(y_est, y, sgrp=s, history=history['train'], queue_len=train_metric_samples)
            
            
        #====================== Validation ======================#
        
        # set the model to evaluation mode
        model.eval()

        # training using all validation samples
        with torch.no_grad():
            for  i, (x, s, y) in enumerate(valid_loader):
                y_est = model(x, s)
                # record the prediction results
                # :: the following function is use to remove zero-padded samples in batch training
                # :: loss metrics are kept in a FIFO queue per latest samples in order to compute statistics
                history['valid'] = loss_fifo(y_est, y, sgrp=s, history=history['valid'], queue_len=len(_valid_dataset))
                
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
def train_cspd(model, train_dataset, valid_dataset, criterion, optimizer, 
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
        train_dataset: List of K element SgrpDataBatch class Pytorch dataloader object
        valid_dataset: List of K element SgrpDataBatch class Pytorch dataloader object
        criterion: Training criterion to be used (eg. criterion = nn.MSELoss())
        optimizer: Training optimizer to be used (eg. optimizer = torch.optim.Adam(model.parameters(), lr = 0.1))
        epochs: Number of training epochs to be used
        batch_size: The batch size to use for batch gradient descent of the output dimension, the input dimension will be setted to zero patching within the SgrpDataBatch object for comparable input size to perform the stacked computation
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
        dataset_model, dataset_test, dataset_train_kfold, dataset_valid_kfold = patitioned_data_object_cspd(x, y, subgroup_col, y_id_col, test_split_ratio, k)
        dataset_train = dataset_train_kfold
        dataset_valid = dataset_valid_kfold
        architecture = [ [2,2,3,2,2] ]   # single branch with 5 layers
        model = cspd(in_feat=10, Subgroups=1, Architecture=architecture, dropout_p=0.3)
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
    
    if batch_size > 1:
        train_dataset.zero_patched = True
        valid_dataset.zero_patched = True
    else:
        train_dataset.zero_patched = False
        valid_dataset.zero_patched = False

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):

        #====================== Training ======================#
        running_loss = 0.0
        epoch_steps = 0

        # training using all training samples
        for i, (x, sgrp, y) in enumerate(train_loader):
            # zero the parameter gradients
            optimizer.zero_grad()

            # set the model to training mode
            model.train()
            
            # forward + backward + optimize
            y_est = model(x, sgrp)
            y_est, y = remove_zero_padded(y_est, y, sgrp)
            loss = criterion(y_est, y)
            loss.backward()
            optimizer.step()
            history['train'] = loss_fifo(y_est, y, sgrp=sgrp, history=history['train'], queue_len=train_metric_samples)

        #====================== Validation ======================#
        
        # set the model to evaluation mode
        model.eval()

        # training using all validation samples
        with torch.no_grad():
            for  i, (x, sgrp, y) in enumerate(valid_loader):
                y_est = model(x, sgrp)
                history['valid'] = loss_fifo(y_est, y, sgrp=sgrp, history=history['valid'], queue_len=len(valid_dataset))
                
    
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


def train_cspd_raytune_cpu_gpu_distributed(config, 
                                           num_in_feat,
                                           num_branch,
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
        result = train_cspd_raytune(config, 
                                    num_in_feat,
                                    num_branch,
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
        result = train_cspd_raytune(config, 
                                    num_in_feat,
                                    num_branch,
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


