import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import confusion_matrix, f1_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import torch.nn.functional as F

import os
import sys
import itertools
import warnings
import filelock

import string
import time
import random

import pdb
from .shared import *

############################ Deep Learning Architecture ############################

class MultiLayerFC(nn.Module):

    # Constructor
    def __init__(self, in_feat, layers, dropout_p=None, act_fn=torch.relu, dtype=torch.double):
        super(MultiLayerFC, self).__init__()
        layers = [in_feat] + layers   # Add input layer
        self.hidden = nn.ModuleList()
        self.out = nn.Linear(layers[-1], 1).type(dtype)
        self.act_fn = act_fn
        self.dropout = nn.Dropout(p=dropout_p)
        # --- Scalable Layers ---
        for input_size, output_size in zip(layers, layers[1:]):
            self.hidden.append(nn.Linear(input_size,output_size).type(dtype))
            
    # Prediction
    def forward(self, x):
        L = len(self.hidden)
        for (l, single_layer) in zip(range(L), self.hidden):
            x = single_layer(x)
            x = self.dropout(self.act_fn(x))
        x = self.out(x)
        return x

        
############################ OHPL Loss ############################

def ohpl(y_true, pred, min_label, max_label, margin=1, ordering_loss_weight=1, loss_bound=1e9, ohpl_norm_order=2, dtype=torch.double):
    '''
    OHPL Hyperplane Loss Function
    (Modified from: https://github.com/ohpl/ohpl/blob/master/OHPLall.ipynb)
    
    Args:
        y_true: Ground Truth of output
        y_pred: Network output (w^T Phi(x)) - NOT CATEGORICAL OUTPUT!!!
        minlabel: Minimum ordinal categorical label of y
        maxlabel: Maximum ordinal categorical label of y
        margin:
        ordering_loss_weight: 
        
    Returns:
        mean_loss: Loss measure

    Raises:
        -

    Author:
        Dr. Calvin Chan
        calvin.chan@bayer.com

    '''
    
    # === HCL: Hyperplane Centroid Loss ===
    # (To ensure hyperplane are ordered by rank)

    pred = pred.type(dtype)
    y_true = y_true.type(dtype)
    ords, idx = torch.unique(y_true, return_inverse=True)
    num_label = ords.shape[0]
    y_true_ohe = F.one_hot(idx,num_classes=num_label)

    # hyperplane intercept term
    yO = torch.transpose(pred.type(dtype),0,1) @ y_true_ohe.type(dtype)
    yc = torch.sum(y_true_ohe, dim=0)
    class_mean = torch.div(yO,yc).type(dtype)

    # relative rank distance between centroids
    min_distance = torch.reshape(ords,(-1,1)) - torch.reshape(ords,(1,-1))
    min_distance = torch.relu(min_distance)

    # keeps min. distance (???)
    keep = torch.minimum(min_distance,torch.ones(min_distance.shape))
    
    # positive mean sample distance between centroids
    centroid_distance = torch.reshape(class_mean,(-1,1)) - torch.reshape(class_mean,(1,-1))
    centroid_distance = torch.relu(centroid_distance)   # zero loss for correct ordering
    centroid_distance = torch.multiply(keep, centroid_distance)

    hp_ordering_loss = torch.sum(torch.relu(min_distance - centroid_distance))

    # === HPL/HPPL: Hyperplane Point Loss ===
    # (To ensure transformation place the point near the correct centroid)
    mean_centroid_of_sample = y_true_ohe.type(dtype) @ torch.reshape(class_mean,(-1,1))

    # --- Limit Edge Case Loss ---
    # No reason to limit distance from edge cases:
    # 1. Positive edge case (max_label) for upper loss
    # 2. Negative edge case (min_label) for lower loss
    upper_bound = (y_true - max_label + 1) * loss_bound   # Select edge case and give a large loss_bound (we want to pull it back in case if it gets too big)
    upper_bound = torch.relu(upper_bound) + margin        # Add margin to non-edge cases
    lower_bound = (-(y_true - min_label) + 1) * loss_bound
    lower_bound = torch.relu(lower_bound) + margin   

    # -- Compute Loss ---
    upper_loss = pred[:,None] - mean_centroid_of_sample
    upper_loss_bounded = torch.relu(upper_loss - upper_bound[:,None])
    lower_loss = -(pred[:,None] - mean_centroid_of_sample)
    lower_loss_bounded = torch.relu(lower_loss - lower_bound[:,None])

    hp_point_loss = torch.mean(upper_loss_bounded + lower_loss_bounded)

        
    # === OHPL ===
    loss = torch.norm(torch.cat([hp_point_loss[None], (ordering_loss_weight * hp_ordering_loss)[None]]), p=ohpl_norm_order)
    
    return loss


############################ Training Procedure ############################

def ohpl_y_class_mean(y):
    '''
    Sample class mean calculation for computing centroid
    The training sample class mean matrix was previously part of the dataloader object.  However, to allow maximum flexibility of 
    random sampling, it was separated out.
    
    Args:
        y: Class label of training samples
    
    Return:
        class_mean: The mean value of each class label for each sample according to their class
    '''
    ohe_encoder = OneHotEncoder(sparse=False, categories='auto')
    y_ohe = ohe_encoder.fit_transform(y)
    y_ohe_inverse = 1/np.sum((y_ohe), axis=0)
    class_mean = (y_ohe * y_ohe_inverse).T
    return class_mean


def ohpl_predict(pred, centroid, min_label, delta=1e-9):
    '''
    OHPL Class Label prediction using training centroid
    The OHPL train the transformation function, but the class label prediction requires using the function as well as the
    centroid computed during training.  The output model only provide the sample projected dimension and distance between
    the model transformed output and the centroid is required to determine class.
    
    Args:
        pred: Model tranformed output at the ordinal hyperplane space
        centroid: Class associated centroid in the ordinal hyperplane space
        min_label: Lowest rank class label
    
    Return:
        y_pred: Predicted class label
        
    Raises:
        -

    Author:
        Dr. Calvin Chan
        calvin.chan@bayer.com

    '''
    y_dist = torch.abs(pred - centroid)
    y_prob = (1/(y_dist+delta))/torch.sum(1/(y_dist+delta),axis=1)[:,None]   # convert distance to probability for cross-entropy computation
    y_pred = torch.argmin(y_dist, axis=1) + min_label
    return y_pred, y_prob


def train_ohpl(model, train_dataset, valid_dataset, min_label, max_label, criterion, optimizer, 
               num_epochs=100, 
               batch_size=2, 
               metric_dict = {'acc': lambda y_est,y: (torch.sum(y_est == y)/torch.tensor(y.shape[0])).item(),
                              'mae': lambda y_est,y: (torch.mean(abs(y_est-y))).item(), 
                              'mze': lambda y_est,y: (torch.mean((torch.abs(y_est-y) > 0).type(torch.double))).item(),
                              'f1-micro':  lambda y_est,y: f1_score(y,y_est,average='micro'),
                              'f1-macro':  lambda y_est,y: f1_score(y,y_est,average='macro'),},
               margin=1,
               ordering_loss_weight=1, 
               loss_bound=1e9,
               ohpl_norm_order=1,
               show_progress=True,
               dtype=torch.double,
               ):
    '''
    Training procedure for OHPL classifier
    
    Author:
        Dr. Calvin Chan
        calvin.chan@bayer.com
    '''
    history = {'train': {}, 'valid': {}}
    metric_output = {}

    # gpu usage
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # parallel gpu usage
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)   # for multiple GPUs
    model.to(device)
    model.apply(initialize_weights)
    
    # initialize dataloader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=lambda x: [ x_ele.to(device) for x_ele in default_collate(x) ])
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=lambda x: [ x_ele.to(device) for x_ele in default_collate(x) ])
    
    for epoch in range(num_epochs):

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
            pred = model(x)
            
            ohpl_loss = criterion(y.squeeze(dim=1),
                                  pred, 
                                  min_label, 
                                  max_label, 
                                  margin, 
                                  ordering_loss_weight, 
                                  loss_bound,
                                  ohpl_norm_order,
                                  dtype)
            ohpl_loss.backward()
            optimizer.step()

        # set the model to evaluation mode
        model.eval()

        #====================== Compute Metrics ======================#
        # model can only be evaluated after finishing the complete dataset for OHPL

        train_pred = torch.tensor([])
        valid_pred = torch.tensor([])
        history['train']['y'] = torch.tensor([])
        history['valid']['y'] = torch.tensor([])
        history['train']['y_est'] = torch.tensor([])
        history['valid']['y_est'] = torch.tensor([])

        with torch.no_grad():
            for  i, (x, y) in enumerate(train_loader):
                # Product network output for a single sample batch
                pred = model(x)
                
                # Collect network output for all samples
                # (The centroid can only be computed using all training samples)
                train_pred = torch.cat([train_pred,pred])
                
                # Collect all output for loss computation
                history['train']['y'] = torch.cat( [history['train']['y'], y.squeeze(dim=1)], dim=0 )

            # Compute centroid 
            y_class_mean = ohpl_y_class_mean(history['train']['y'].reshape(-1,1))
            centroid = torch.reshape( torch.tensor(y_class_mean @ train_pred.numpy()), [1,-1] )
            
            # Predict for all training samples
            y_est_train, train_prob = ohpl_predict(train_pred, centroid, min_label)
            
            # Collect all estimated output for loss computation
            history['train']['y_est'] = torch.cat( [history['train']['y_est'], y_est_train], dim=0 )
            
            # !!! DEBUG: Product data dimension as RayTune metrics !!!
#             metric_output['y_class_mean.shape'] = y_class_mean.shape   # (4, 128)
#             metric_output['centroid.shape'] = centroid.shape           # torch.Size([1, 4])
#             metric_output['train_pred.shape'] = train_pred.shape       # torch.Size([128, 1])
            
            for  i, (x, y) in enumerate(valid_loader):
                pred = model(x)
                valid_pred = torch.cat([valid_pred,pred])
                history['valid']['y'] = torch.cat( (history['valid']['y'], y.squeeze(dim=1)), dim=0 )
            y_est_valid, valid_prob = ohpl_predict(valid_pred, centroid, min_label)
            history['valid']['y_est'] = torch.cat( [history['valid']['y_est'], y_est_valid], dim=0 )

                        
        # compute loss metrics based on y_pred & y_true
        metric_labels = []
        for metric in metric_dict.keys():
            for dataset in history.keys():
                metric_label = '_'.join([dataset,metric])
                metric_output[metric_label] = metric_dict[metric](history[dataset]['y_est'],history[dataset]['y'])
                metric_labels.append(metric_label)

        # cross-entropy-loss metric requires probability matrix
        # (not using y_pred & y_true for computation, therefore need to separate out)
        cross_entropy_loss = nn.CrossEntropyLoss()
        metric_labels.append("train_cross-entropy")
        metric_labels.append("valid_cross-entropy")
        cel_train = cross_entropy_loss(train_prob, history['train']['y'].type(torch.int64) - min_label)
        cel_valid = cross_entropy_loss(valid_prob, history['valid']['y'].type(torch.int64) - min_label)
        metric_output['train_cross-entropy'] = cel_train
        metric_output['valid_cross-entropy'] = cel_valid
        
        # display metrics
        if show_progress:
            print(f"[Epoch: { epoch+1 }]", end=" " )
            print(f"OHPL Loss: {ohpl_loss}")
            for metric_label in metric_labels:
                print(f"{metric_label}: {metric_output[metric_label].item():.3f},", end=" ")
            print(f"")

    return (centroid, history)


def train_ohpl_raytune_cpu_gpu_distributed(config, 
                                           num_in_feat,
                                           train_dataset, 
                                           valid_dataset,
                                           criterion=ohpl,
                                           checkpoint_dir=None, 
                                           num_epochs=100, 
                                           metric_dict = {'acc': lambda y_est,y: (torch.sum(y_est == y)/torch.tensor(y.shape[0])).item(),
                                                          'mae': lambda y_est,y: (torch.mean(abs(y_est-y))).item(), 
                                                          'mze': lambda y_est,y: (torch.mean((torch.abs(y_est-y) > 0).type(torch.double))).item(),
                                                          'f1-micro':  lambda y_est,y: f1_score(y,y_est,average='micro'),
                                                          'f1-macro':  lambda y_est,y: f1_score(y,y_est,average='macro'),},
                                           ohpl_norm_order=1,
                                           dtype=torch.double,
                                           force_cpu=False
                                           ):
    
    '''
    CPU/GPU Distributed Wrapper Function for Training procedure for OHPL classifier
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
        result = train_ohpl_raytune(config, 
                                    num_in_feat,
                                    train_dataset, 
                                    valid_dataset,
                                    criterion=ohpl,
                                    checkpoint_dir=None, 
                                    num_epochs=num_epochs, 
                                    metric_dict = {'acc': lambda y_est,y: (torch.sum(y_est == y)/torch.tensor(y.shape[0])).item(),
                                                   'mae': lambda y_est,y: (torch.mean(abs(y_est-y))).item(), 
                                                   'mze': lambda y_est,y: (torch.mean((torch.abs(y_est-y) > 0).type(torch.double))).item(),
                                                   'f1-micro':  lambda y_est,y: f1_score(y,y_est,average='micro'),
                                                   'f1-macro':  lambda y_est,y: f1_score(y,y_est,average='macro'),},
                                    ohpl_norm_order=1,
                                    dtype=torch.double,
                                    force_cpu=False
                                    )
                            
    except filelock.Timeout:
        # If the lock is acquired, you can just use CPU, and disable GPU access.
        result = train_ohpl_raytune(config, 
                                    num_in_feat,
                                    train_dataset, 
                                    valid_dataset,
                                    criterion=ohpl,
                                    checkpoint_dir=None, 
                                    num_epochs=num_epochs, 
                                    metric_dict = {'acc': lambda y_est,y: (torch.sum(y_est == y)/torch.tensor(y.shape[0])).item(),
                                                   'mae': lambda y_est,y: (torch.mean(abs(y_est-y))).item(), 
                                                   'mze': lambda y_est,y: (torch.mean((torch.abs(y_est-y) > 0).type(torch.double))).item(),
                                                   'f1-micro':  lambda y_est,y: f1_score(y,y_est,average='micro'),
                                                   'f1-macro':  lambda y_est,y: f1_score(y,y_est,average='macro'),},
                                    ohpl_norm_order=1,
                                    dtype=torch.double,
                                    force_cpu=True
                                    )
    finally:
        # Release the lock after training is done.
        a.release()
    return result


def train_ohpl_raytune(config, 
                       num_in_feat,
                       train_dataset, 
                       valid_dataset,
                       criterion=ohpl,
                       checkpoint_dir=None, 
                       num_epochs=100, 
                       metric_dict = {'acc': lambda y_est,y: (torch.sum(y_est == y)/torch.tensor(y.shape[0])).item(),
                                      'mae': lambda y_est,y: (torch.mean(abs(y_est-y))).item(), 
                                      'mze': lambda y_est,y: (torch.mean((torch.abs(y_est-y) > 0).type(torch.double))).item(),
                                      'f1-micro':  lambda y_est,y: f1_score(y,y_est,average='micro'),
                                      'f1-macro':  lambda y_est,y: f1_score(y,y_est,average='macro'),},
                       ohpl_norm_order=1,
                       dtype=torch.double,
                       force_cpu=False
                       ):
    '''
    Hyperparameter Testing for OHPL classifier

    Author:
        Dr. Calvin Chan
        calvin.chan@bayer.com
    '''

    # gpu usage
    if not force_cpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"

    #====================== Ray Tune Parameters Setup ======================#

    if 'dropout_p' in config.keys():
        _dropout_p = config['dropout_p']
    else:
        _dropout_p = 0

    if 'batch_size' in config.keys():
        _batch_size = config['batch_size']
    else:
        _batch_size = 1
        
    if 'margin' in config.keys():
        _margin = config['margin']
    else:
        _margin = 1
        
    if 'ordering_loss_weight' in config.keys():
        _ordering_loss_weight = config['ordering_loss_weight']
    else:
        _ordering_loss_weight = 1
        
    if 'loss_bound' in config.keys():
        _loss_bound = config['loss_bound']
    else:
        _loss_bound = 1e9
        
    if (type(train_dataset) is list) and (type(valid_dataset) is list) and ('k' in config.keys()):
        _train_dataset = train_dataset[config['k']]
        _valid_dataset = valid_dataset[config['k']]
    else:
        _train_dataset = train_dataset
        _valid_dataset = valid_dataset
    
    train_loader = torch.utils.data.DataLoader(dataset=_train_dataset, batch_size=_batch_size, shuffle=True, 
                                               collate_fn=lambda x: [ x_ele.to(device) for x_ele in default_collate(x) ] )
    valid_loader = torch.utils.data.DataLoader(dataset=_valid_dataset, batch_size=_batch_size, shuffle=True,
                                               collate_fn=lambda x: [ x_ele.to(device) for x_ele in default_collate(x) ] )

    y_col_index = 0
    min_label = _train_dataset.y_min[y_col_index]
    max_label = _train_dataset.y_max[y_col_index]
    
    # initialize ANN architecture
    model = MultiLayerFC(in_feat = num_in_feat,
                         layers = config['h_fc'],
                         dropout_p = _dropout_p,
                         act_fn = torch.relu).to(device)
    model.apply(initialize_weights)
    
    # optimizer is controlled by ray tune hyperparameter
    optimizer = torch.optim.Adam(model.parameters(), lr = config["lr"])
    
    for epoch in range(num_epochs):

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
            pred = model(x)

            ohpl_loss = criterion(y.squeeze(dim=1), 
                                  pred, 
                                  min_label, 
                                  max_label, 
                                  _margin, 
                                  _ordering_loss_weight, 
                                  _loss_bound,
                                  ohpl_norm_order,
                                  dtype)
            ohpl_loss.backward()
            optimizer.step()

        # set the model to evaluation mode
        model.eval()

        #====================== Compute Metrics ======================#
        # model can only be evaluated after finishing the complete dataset for OHPL

        train_pred = torch.tensor([])
        valid_pred = torch.tensor([])
        history = {'train': {}, 'valid': {}}
        history['train']['y'] = torch.tensor([])
        history['valid']['y'] = torch.tensor([])
        history['train']['y_est'] = torch.tensor([])
        history['valid']['y_est'] = torch.tensor([])
        metric_output = {}
        
        with torch.no_grad():
            for  i, (x, y) in enumerate(train_loader):
                # Product network output for a single sample batch
                pred = model(x)
                
                # Collect network output for all samples
                # (The centroid can only be computed using all training samples)
                train_pred = torch.cat([train_pred,pred])
                
                # Collect all output for loss computation
                history['train']['y'] = torch.cat( [history['train']['y'], y.squeeze(dim=1)], dim=0 )

            # Compute centroid 
            y_class_mean = ohpl_y_class_mean(history['train']['y'].reshape(-1,1))
            centroid = torch.reshape( torch.tensor(y_class_mean @ train_pred.numpy()), [1,-1] )
            
            # Predict for all training samples
            y_est_train, train_prob = ohpl_predict(train_pred, centroid, min_label)
#             
            # Collect all estimated output for loss computation
            history['train']['y_est'] = torch.cat( [history['train']['y_est'], y_est_train], dim=0 )
            
            # !!! DEBUG: Product data dimension as RayTune metrics !!!
#             metric_output['y_class_mean.shape'] = y_class_mean.shape   # (4, 128)
#             metric_output['centroid.shape'] = centroid.shape           # torch.Size([1, 4])
#             metric_output['train_pred.shape'] = train_pred.shape       # torch.Size([128, 1])
            
            for  i, (x, y) in enumerate(valid_loader):
                pred = model(x)
                valid_pred = torch.cat([valid_pred,pred])
                history['valid']['y'] = torch.cat( (history['valid']['y'], y.squeeze(dim=1)), dim=0 )
            y_est_valid, valid_prob = ohpl_predict(valid_pred, centroid, min_label)
            history['valid']['y_est'] = torch.cat( [history['valid']['y_est'], y_est_valid], dim=0 )

        for metric in metric_dict.keys():
            for dataset in history.keys():
                metric_label = '_'.join([dataset,metric])
                metric_output[metric_label] = metric_dict[metric](history[dataset]['y_est'],history[dataset]['y'])#.item()

        # ohpl metric append to output
        metric_output['ohpl'] = ohpl_loss.item()
                
        # cross-entropy-loss metric requires probability matrix
        # (not using y_pred & y_true for computation, therefore need to separate out)
        cross_entropy_loss = nn.CrossEntropyLoss()
        cel_train = cross_entropy_loss(train_prob, history['train']['y'].type(torch.int64) - min_label)
        cel_valid = cross_entropy_loss(valid_prob, history['valid']['y'].type(torch.int64) - min_label)
        metric_output['train_cross-entropy'] = cel_train.item()
        metric_output['valid_cross-entropy'] = cel_valid.item()
        
        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and will potentially be passed as the `checkpoint_dir`
        # parameter in future iterations.
        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save( (model.state_dict(), optimizer.state_dict(), centroid), path )

          # !!! DEBUG: Product data dimension as RayTune metrics !!!
#         metric_output['y-dim'] = y.shape
#         metric_output['y_est-dim_train'] = y_est_train.shape
#         metric_output['y_est-dim_train_his'] = history['train']['y_est'].shape
#         metric_output['y_est-dim_valid'] = y_est_valid.shape
#         metric_output['y_est-dim_valid_his'] = history['valid']['y_est'].shape
#         metric_output['pred-dim'] = pred.shape

        tune.report(**metric_output)
        
        
