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
from random import uniform
from gekko import GEKKO

import os
import itertools
import warnings
import filelock

import math
import string
import time
import random

############################ General Functions ############################

def select_ind(ls,ind):
    return [ ls[i] for i in ind.tolist() ]


def unique_list(ls_of_ls):
    '''
    Return the unique list in a list of list
    
    Args:
        ls_of_ls: List of list (eg. [[1,2,3],[1,3,2],[1,2,3]])

    Returns:
        unique_ls: Unique list within the input list (eg. [[1,3,2],[1,2,3]])

    Author:
        Dr. Calvin Chan
        calvin.chan@bayer.com
    '''
    unique_ls = [list(ls_out) for ls_out in set(tuple(ls) for ls in ls_of_ls)]
    return unique_ls


############################ Deep learning General ############################

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)

        
############################ Data Handling ############################

def convert_multidimensional_labels(df,col):
    '''
    Convert Multiple Column Label into Single Column
    
    Args:
        df: A pandas dataframe with row as samples, and column as N-dimensional subgroup to be encoded.
        col: Column name of the combined column
        
    Returns:
        df: A pandas dataframe with new label column

    Raises:
        -

    Author:
        Dr. Calvin Chan
        calvin.chan@bayer.com
    '''
    if df.shape[1] == 1:
        df = pd.concat([df,df],axis=1)
        df.columns = [df.columns[0],col]
    else:
        df[col] = tuple(labels.values.tolist())
        df[col] = labels[col].apply(lambda x: ','.join([str(c) for c in x ]))
    return(df)


def combine_multidimensional_ohe(s):
    '''
    One-Hot-Encoding (OHE) based on joint label of multiple columns
    The default OHE feature of Pandas and sklearn takes each column as independent OHE. 
    This function uses the 2D unique label combination as a single dimension for OHE.
    
    Args:
        s: A pandas dataframe with row as samples, and column as N-dimensional subgroup to be encoded.

    Returns:
        s_ohe: A pandas dataframe with N-D OHE
        conversion_table: The conversion table for N-D OHE

    Raises:
        -

    Author:
        Dr. Calvin Chan
        calvin.chan@bayer.com
    '''
    unique_labels = [ sorted(s[name].unique().tolist()) for name in s.columns.tolist() ]
    multidimensional_labels = [*itertools.product(*unique_labels)]
    labels = pd.DataFrame(multidimensional_labels, columns=s.columns.tolist())
    labels = convert_multidimensional_labels(labels,'sgrp')
    conversion_table = pd.get_dummies(labels, columns=['sgrp'])
    s_ohe = pd.merge(s,conversion_table,on=s.columns.tolist(),how='left').drop(s.columns.tolist(),axis=1)
    return(s_ohe,conversion_table)


def model_test_split(*args, id_col=None, test_ratio=0.2, random_state=25, report_id=False):
    
    '''
    Split the dataset into modeling and test set
    
    This function is to encapsulate the variying input feature size given the grouping by id_col,
    and this decompose the one-hot-encoding column into a separate feature set to be used in the
    deep learning model as separate input.
    
    Args:
        *args:
            x: A pandas dataframe with row as samples, and column as ID and feature type
            y: A pandas dataframe with row as samples, and column as output
        ohe_col: A list of column names indicating the one-hot-encoding columns in x
        id_col: Column name of the grouping column to be converted to one-hot-encoding
        test_size: The split ratio of the test set
        random_state: Random seed use by the `sklearn.model_selection.train_test_split` function
        retain_df: If this is 'True' and the input 'args' are dataframes, do not convert them to list of single row dataframe

    Returns:
        x_model, x_test: List of numpy matrix as model/test data split with from commond id_col labels of x and y
        s_model, s_test: List of numpy matrix as model/test data split with from commond id_col labels of x and y
        y_model, y_test: List of numpy matrix as model/test data split with from commond id_col labels of x and y

    Raises:
        Warning when the labels in id_col of x and y do not match
        
    Author:
        Dr. Calvin Chan
        calvin.chan@bayer.com
    '''

    if id_col is not None:

        inds = []
        data = []
        for arg in args:
            (ind,dat) = zip(*list(arg.groupby(id_col)))
            inds.append(ind)
            data.append(dat)

        # Determine of ID entry is missing from any of the input dataset
        id_match_flag = (len(set.intersection(*[set(ind) for ind in inds])) == len(set.union(*[set(ind) for ind in inds])))
        if not id_match_flag:
            warnings.warn("Unmatch ID entries in one or more data inputs (eg. x, y)!")

        # Extract Common ID from x, s, y Samples
        select_ids = list(set.intersection(*[set(ind) for ind in inds]))

        # Split dataframes into sample list
        # (multi-resolution: each list element contains multiple x and single y based on id_col)
        dataset = []
        for i, dat in enumerate(data):
            dataset.append([ dat[inds[i].index(single_id)].drop(id_col,axis=1) for single_id in select_ids ])

    else:
        # Determine index labels in each input dataset is the same
        dataset_indices = [ list(dataset.index) for dataset in args ]
        select_ids = unique_list(dataset_indices)
        id_match_flag = (len(select_ids) == 1)
        assert id_match_flag, "Unmatch length in one or more data inputs (eg. x, y)!"
        select_ids = select_ids[0]
        
        # Split dataframes into sample list
        # (equal resolution: each list element contains one row in both x and y)
        dataset = []
        for i, dat in enumerate(args):
            dataset.append([ dat.loc[[single_id]] for single_id in select_ids ])

    # Including index as one of the splitting dataset
    assert (test_ratio*len(dataset) >= 1), "Number of samples resulting from ratio must be larger than 1 sample!"
    dataset = dataset + [select_ids]
    out = train_test_split(*dataset, test_size=test_ratio, random_state=random_state)
    split_ids = out[-2:]
    out = out[0:-2]

    if report_id:
        return(out, split_ids)
    else:
        return(out)

    
def get_k_fold_indices(n_samples, k=5, shuffle=False):
    '''
    Drawing sample indices for K-Fold
    
    Args:
        samples: Number of samples in the dataset
        shuffle: Shuffling of samples

    Returns:
        kfold_train_ind: Indices for training set
        kfold_valid_ind: Indices for validation set

    Raises:
        -

    Author:
        Dr. Calvin Chan
        calvin.chan@bayer.com
    '''
    kfold = KFold(n_splits=k, shuffle=shuffle).split([*range(n_samples)])
    i, kfold_ind = zip(*[*enumerate(kfold)])   # Expand the index obtained by the K-Fold function
    kfold_train_ind, kfold_valid_ind = zip(*kfold_ind)
    return(kfold_train_ind, kfold_valid_ind)


############################ Pytorch Data Object ############################    

class NumericData(Dataset):
    def __init__(self, x, y, transform=None, dtype=torch.double, sample_ids=None):
        assert (len(y) == len(x)), "Number of x and y samples do not match!"
        self.len = len(y)
        self.transform = transform
        self.sample_ids = sample_ids
        self.return_id = False
        
        self.x, self.x_col, self.x_min, self.x_max = self._format_dataset(x, dtype)
        self.y, self.y_col, self.y_min, self.y_max = self._format_dataset(y, dtype)

    def __getitem__(self, index):
        
        sample = [self.x[index],
                  self.y[index]]
        if self.transform:
            sample = self.transform(sample)

        if self.return_id:
            # id tracking for error investigation
            return sample, index
        else:
            # return sample for training
            return sample
    
    def __len__(self):
        return self.len
    
    def _format_dataset(self, d, dtype):
        
        if type(d) == pd.core.frame.DataFrame:
            # check to make sure that the sample_ids are the same as dataframe row index if sample_ids exist
            if self.sample_ids is not None:
                assert (len(unique_list([list(d.index),self.sample_ids])) == 1), "Input data rowname/index not equal to sample_ids!"
            else:
                self.sample_ids = list(d.index)
            
            # extract column names
            colname = d.columns
            
            # get y-min/max (required for OHPL)
            d_max = d.max(axis=0).tolist()          
            d_min = d.min(axis=0).tolist()
            
            # convert d to list of single row dataframe
            d = [ d.loc[[ind]] for ind in d.index ]
            
        else:
            # extract column names
            colname = d[0].columns

            # get y-min/max (required for OHPL)
            d_max = pd.concat(d).max().tolist()
            d_min = pd.concat(d).min().tolist()
            
        # convert dataframe to list of a single row tensor
        out = self._sample_type_convert(d, dtype)

        return out, colname, d_min, d_max
        
    def _sample_type_convert(self, samples, dtype):
        # since the input samples are list of single-row-dataframe, with dimension of 1 x Features
        # to convert them into tensors, the row dimension is removed.
        samples_out = [ torch.tensor(sample_ele.iloc[0]).type(dtype) for sample_ele in samples ]
        return samples_out
        
    def _sample_type_convert(self, samples, dtype):
        # since the input samples are list of single-row-dataframe, with dimension of 1 x Features
        # to convert them into tensors, the row dimension is removed.
        samples_out = [ torch.tensor(sample_ele.iloc[0]).type(dtype) for sample_ele in samples ]
        return samples_out     
    
    
############################ Data Partitioning ############################

def patitioned_data_object_numeric(x, y, test_split_ratio, k, random_state=25, kfold_ind=False):
    
    
    assert len(x) == len(y), warnings.warn("Unmatch number of data samples with x and y!")
    
    # K-Fold Index Sampling
    [kfold_train_ind, kfold_valid_ind] = get_k_fold_indices(n_samples=len(y), k=k, shuffle=False)   # Shuffle is NOT needed, since the samples were shuffled in the model/test split
    
    # Test set is zero size
    zero_ratio_flag = True if (len(x) * test_split_ratio) < 1 else False
    if zero_ratio_flag:
        warnings.warn("Sample with given ratio is less than 1, the entire set of data is used for modeling!")
        x_model = x
        y_model = y
        samples_id_model = x.index.values.tolist()
        samples_id_test = []
        x_test = pd.DataFrame(data=None, columns=x.columns)
        y_test = pd.DataFrame(data=None, columns=y.columns)
        

        # Create K-set of datasets for Pytorch data loader
        dataset_train_kfold = [ NumericData(x_model.loc[fold_ind], 
                                            y_model.loc[fold_ind],
                                            sample_ids = select_ind(samples_id_model,fold_ind))
                                               for fold_ind in kfold_train_ind ]
        dataset_valid_kfold = [ NumericData(x_model.loc[fold_ind], 
                                            y_model.loc[fold_ind],
                                            sample_ids = select_ind(samples_id_model,fold_ind))
                                               for fold_ind in kfold_valid_ind ]
    # Test set non-zero size
    else:
        # Model/Test Splitting
        (x_model, x_test, 
         y_model, y_test), (samples_id_model, samples_id_test) = model_test_split(x, y, 
                                                                 test_ratio=test_split_ratio, 
                                                                 report_id=True,
                                                                 random_state=random_state)
    
        # Create K-set of datasets for Pytorch data loader
        dataset_train_kfold = [ NumericData(select_ind(x_model,fold_ind), 
                                            select_ind(y_model,fold_ind),
                                            sample_ids = select_ind(samples_id_model, fold_ind)) 
                                               for fold_ind in kfold_train_ind ]
        dataset_valid_kfold = [ NumericData(select_ind(x_model,fold_ind), 
                                            select_ind(y_model,fold_ind),
                                            sample_ids = select_ind(samples_id_model, fold_ind)) 
                                               for fold_ind in kfold_valid_ind ]

    # Create dataset for modeling and testing
    dataset_model = NumericData(x_model, y_model, sample_ids = samples_id_model)
    dataset_test = NumericData(x_test, y_test, sample_ids = samples_id_test)
    
    if kfold_ind:
        return dataset_model, dataset_test, dataset_train_kfold, dataset_valid_kfold, kfold_train_ind, kfold_valid_ind
    else:
        return dataset_model, dataset_test, dataset_train_kfold, dataset_valid_kfold


############################ Hyperparameter Sampling ############################

##############################
### Fixed Neurons Sampling ###
##############################

def integer_partitions(n_ele, n_min=1, max_dim=None, recursion_level=1):
    '''
    Fast Integer Partitioning
    Dividing a single integer into a list of integer that sums up to the given number
    
    Args:
        num_ele: Total number of elements to be distributed
        n_min: Minimum number of elements per output dimension

    Returns:
        Iterator as list of elements splitted into multiple dimensions
        
    Original Source :
    (Modification made to speed up by skpping recurrsion exceed max_dim)
        https://stackoverflow.com/questions/10035752/elegant-python-code-for-integer-partitioning
    
    Author:
        Dr. Calvin Chan
        calvin.chan@bayer.com
    '''
    if (max_dim is not None) and (recursion_level > max_dim):
        yield None
    else:
        yield (n_ele,)
        for i in range(n_min, n_ele//2 + 1):
            for p in integer_partitions(n_ele-i, i, max_dim, recursion_level+1):
                if p is not None:
                    yield (i,) + p
                elif recursion_level != 1:
                    yield None

def split_sampling(num_ele, num_layers=None, n_min=1, n_max=None, n_samples=1, prepend=[], postpend=[], single_sample=False):
    '''
    Randomly split the elements into multiple dimensions
    This is use for neuron sampling the number of elements and layer for multibranch neural network
    
    Args:
        num_ele: Total number of elements to be distributed
        n_min: Minimum number of elements per output dimension
        n_max: Maximum number of elements per output dimension
        num_layers: Number of layers to distribute the element, random dimensions will be given with None given

    Returns:
        sample: List of elements splitted into multiple dimensions
        
    Raises:
        -
        
    Example:
        >>> split_sampling(14, n_min=2, num_layers=4)
        [2, 5, 4, 3]
        
    Author:
        Dr. Calvin Chan
        calvin.chan@bayer.com
    '''
    # !!! DEBUG !!!
    # print(f"num_ele: {num_ele}; n_min: {n_min}; num_layers: {num_layers}")
    
    # Generate the Integer Partitions
    splits = integer_partitions(num_ele, n_min=n_min, max_dim=num_layers)
    if n_max is not None:
        splits = [ split for split in splits if max(split) <= n_max ]
    if num_layers is not None:
        splits = [ split for split in splits if len(list(split)) == num_layers ]
    else:
        splits = [ split for split in splits ]
    
    # Filter with Number of Output Dimension
    splits_perm = [list(set(itertools.permutations(split))) for split in splits ]
    unique_splits_perm = list(itertools.chain.from_iterable(splits_perm))
        
    # Randomly Sample one of the permutation
    if n_samples <= len(unique_splits_perm):
        sample = list([ prepend+list(sample)+postpend for sample in random.sample(unique_splits_perm, k=n_samples)])
    else:
        sample = list([ prepend+list(sample)+postpend for sample in random.choices(unique_splits_perm, k=n_samples)])
    if single_sample:
        sample = sample[0]
    
    return(sample)                    

def compile_architecture_table(h_total_min, h_total_max, h_total_step,
                               h_subgroup_dim, h_subgroup_min_neuron_per_sgrp, h_subgroup_max_neuron_per_sgrp, h_subgroup_n_samples,
                               h_branch_min_neuron_per_layer, h_branch_max_neuron_per_layer, h_branch_n_samples):
    '''
    Generate Architecture Table for the Neural Network Architecture
    
    Args:
        h_total_min, h_total_max, h_total_step: Equally spaced ampling criteria for total number of neuron
        subgroup_dim: Subgroup output dimension (# subgroups)
        h_subgroup_min_neuron_per_sgrp: Minimum total number of neuron per subgroup branch
        h_subgroup_max_neuron_per_sgrp: Maximum total number of neuron per subgroup branch
        h_subgroup_n_samples: Total number of sample for subgroup distribution (from splitted neuron distribution)
        h_branch_min_neuron_per_layer: Minimum total number of neuron per layer
        h_branch_max_neuron_per_layer: Maximum total number of neuron per layer
        h_branch_n_samples: Total number of sample for branch distriubtion (from splitted neuron distribution)

    Returns:
        Pandas Dataframe with each row containing one random neural network architecture sample with the corresponding architecture information in each column.
            
    Author:
        Dr. Calvin Chan
        calvin.chan@bayer.com
    '''
    # Sampling the Architecture given the criteria
    h_total = [*range(h_total_min, h_total_max+1, h_total_step)]
    h_subgroup = [ split_sampling(num_ele = n_neuron, 
                             n_min = h_subgroup_min_neuron_per_sgrp, 
                             n_max = h_subgroup_max_neuron_per_sgrp,
                             n_samples = h_subgroup_n_samples, 
                             num_layers = h_subgroup_dim) for n_neuron in h_total ]
    h_branch = [[[ split_sampling(num_ele = n_subgroup, 
                              num_layers = None,
                              n_min = h_branch_min_neuron_per_layer,
                              n_max = h_branch_max_neuron_per_layer,
                              n_samples = h_branch_n_samples) 
                for n_subgroup in sample ] for sample in total ] for total in h_subgroup ]
    
    # Compile Random Architecture Configuration Table (Table index is used as grid search for hyperparameter tunning)
    total_table = pd.DataFrame({'total_id': [*range(len(h_total))], 'total': h_total})
    
    subgroup_table = pd.DataFrame(columns=['total_id', 'subgroup_id', 'subgroup'])
    for total_id, total in enumerate(h_subgroup):
        for subgroup_id, subgroup in enumerate(total):
                subgroup_table.loc[len(subgroup_table)] = [total_id, subgroup_id, subgroup]
                
    sample_table = pd.DataFrame(columns=['total_id', 'subgroup_id', 'branch_id', 'sample_id', 'h_branch'])
    for total_id, total in enumerate(h_branch):
        for subgroup_id, subgroup in enumerate(total):
            for branch_id, branch in enumerate(subgroup):
                for sample_id, sample in enumerate(branch):
                    sample_table.loc[len(sample_table)] = [total_id, subgroup_id, branch_id, sample_id, sample]
                    
    architecture_table = sample_table.groupby(["total_id","subgroup_id","sample_id"])["h_branch"].apply(list).reset_index()
    architecture_table = pd.merge(architecture_table, subgroup_table, how="left", on=["total_id", "subgroup_id"])
    architecture_table = pd.merge(architecture_table, total_table, how="left", on=["total_id"])
    
    return(architecture_table)


#################################
### Fixed Parameters Sampling ###
#################################

def fc_num_params(H):
    '''
    Compute the total number of parameters in a fully connected neural network
    
    Args:
        H: List of neurons at each layer

    Returns:
        N: Number of parameters in the fully connected network
        
    Raises:
        -
        
    Example:
        >>> fc_num_params([5, 85, 7, 43, 1])
        1500
        
    Author:
        Dr. Calvin Chan
        calvin.chan@bayer.com
    '''

    N = 0
    for i in range(len(H)-1):
        N += H[i+1] * (H[i] + 1)
    return(N)

def parameters_sampling(num_params,
                        num_layers,
                        in_dim,
                        out_dim=1, 
                        n_min=1, 
                        n_max=None, 
                        n_samples=1, 
                        include_inout=True,
                        single_sample=False,
                        max_trials=1000):
    '''
    Randomly sampling DNN architecture based on give number of total parameters.
    This is use for neuron architecture sampling based on the same number of parameters given.
    
    Args:
        num_params: Total number of parameters to be distributed
        num_layers: Total number of layers
        n_min:      Minimum number of neurons per layer
        n_max:      Maximum number of neurons per layer
        in_dim:     Number of neurons at the input layer
        out_dim:    Number of neurons at the output layer
        n_samples:  Number of architecture samples to return (maximum number of samples return if there are less than demanded)
        include_inout: Flag indicate whether to include input and output layer neurons with samples
        max_trials: Maximum number of randomized trial for solution sampling if not enough samples found

    Returns:
        sample: List of elements splitted into multiple dimensions
        
    Raises:
        -
        
    Example:
        >>> parameters_sampling(num_params=500, 
                                num_layers=5,
                                in_dim=5,
                                out_dim=1, 
                                n_min=1, 
                                n_max=None, 
                                n_samples=10, 
                                include_inout=True,
                                single_sample=False,
                                max_trials=1000)
        [[5, 5, 11, 31, 1],
         [5, 6, 7, 46, 1],
         [5, 18, 9, 20, 1],
         [5, 19, 3, 65, 1],
         [5, 29, 5, 25, 1],
         [5, 29, 9, 5, 1],
         [5, 33, 7, 7, 1],
         [5, 49, 3, 11, 1],
         [5, 54, 1, 40, 1],
         [5, 63, 1, 19, 1]]
        
    Author:
        Dr. Calvin Chan
        calvin.chan@bayer.com
    '''
    
    samples = np.empty([0,num_layers],dtype=int)
    
    for trial in range(max_trials):
        # initialize MINLP (Mixed-Integer Nonlinear Programming Model)
        model = GEKKO()

        # Setup variables to be optimized
        H = [in_dim]
        for i in range(1,num_layers-1):
            # Randomize initial variables for diverse samples
            H.append(model.Var(value=round(uniform(H_min,H_max)), integer=True, lb=H_min, ub=H_max))
        H.append(out_dim)

        # Solving for Optimal Values
        model.Minimize(abs(num_params - n_params(H)))
        model.options.SOLVER = 1
        model.solver_options = ['minlp_maximum_iterations 10000',
                                'minlp_branch_method 3',
                                'minlp_max_iter_with_int_sol 500']
        model.solve(disp=False)
        
        H = np.hstack(H).astype(int)
        H = H if include_inout else H[1:-1]

        samples = np.unique(np.vstack((samples, H)), axis=0)

        if samples.shape[0] == n_samples:
            break

    # Convert from Numpy Array back to List
    samples = samples.tolist()
            
    return(samples)


############################ Loss Metrics ############################

def compute_iqr(e):
    '''
    Compute Loss IQR
    
    Args:
        e: Error/Loss

    Returns:
        iqr: Interquartile range of the error

    Raises:
        -

    Example:
        

    Author:
        Dr. Calvin Chan
        calvin.chan@bayer.com
    '''
    q75 = torch.quantile(e, 0.75)
    q25 = torch.quantile(e, 0.25)
    iqr = q75 - q25
    return iqr


def compute_l1_iqr(y_est, y):
    '''
    Compute L1 Loss IQR
    
    Args:
        y_est: Model prediction output
        y: Training data output ground truth

    Returns:
        iqr: Interquartile range of the error

    Raises:
        -

    Example:
        

    Author:
        Dr. Calvin Chan
        calvin.chan@bayer.com
    '''
    e = torch.abs(y_est - y)
    q75 = torch.quantile(e, 0.75)
    q25 = torch.quantile(e, 0.25)
    iqr = q75 - q25
    return iqr


def compute_mape_iqr(y_est, y):
    '''
    Compute Error IQR
    
    Args:
        y_est: Model prediction output
        y: Training data output ground truth

    Returns:
        iqr: Interquartile range of the error

    Raises:
        -

    Example:
        

    Author:
        Dr. Calvin Chan
        calvin.chan@bayer.com
    '''
    e = torch.abs(y_est - y)
    q75 = torch.quantile(e, 0.75)
    q25 = torch.quantile(e, 0.25)
    iqr = q75 - q25
    return iqr


def loss_fifo(y_est, y, sgrp=None, history=None, queue_len=1000):
    '''
    Record Loss of Output Data
    Due to the variable input resolution, zero padding is required for batch gradient decent for cspd algorithm.  Therefore, the zero padded batches could introduce a bias in the loss metric computation.  To avoid this problem, the zero padded data with all zeros for the subgroup indicator is used to remove these entries during error computation.
    
    Args:
        y_est: Model prediction output
        y: Training data output ground truth
        sgrp: Subgrouping one-hot-encoded matrix for the batch data (B x O x S matrix, where B is batch size, O is output dimensions, S is number of subgroups)
        queue_len: Maximum records to be stored in the history queue

    Returns:
        history: Dictionary of output to be reported, each dictionary element is a numpy array as a queue containing the history of past results.

    Raises:
        -

    Example:
        

    Author:
        Dr. Calvin Chan
        calvin.chan@bayer.com
    '''
       
    # remove zero-padded cases
    if sgrp is not None:
        y_est, y, num_pts = remove_zero_padded(y_est, y, sgrp, return_length=True)
    else:
        assert y_est.shape[0] == y.shape[0], "y and y_est has different shape"
        num_pts = y_est.shape[0]
    
    num_pts = min(num_pts, queue_len)   # if queue is smaller than the number of results, truncate the front
    y_est = y_est[-num_pts:]
    y = y[-num_pts:]
        
    # managing the results FIFO queue
    # :: push new sample and remove older samples
    # :: keep the y, y_est in a FIFO for computing statistics
    if history is None or len(history) == 0:
        # initialize for the queue
        history = {'y': y, 'y_est': y_est}
    elif len(history['y']) < queue_len:
        # insert y into non-empty queue and trim data extended beyond queue size
        history['y'] = torch.cat( (history['y'], y), dim=0)[-queue_len:]
        history['y_est'] = torch.cat( (history['y_est'], y_est), dim=0)[-queue_len:]
    else:
        # shift element and replace (push on FIFO)
        history['y'] = torch.roll(history['y'], -num_pts, dims=0)
        history['y'][-num_pts:] = y
        history['y_est'] = torch.roll(history['y_est'], -num_pts, dims=0)
        history['y_est'][-num_pts:] = y_est
    
    return(history)


############################ Results Analysis ############################

def convert_nested_numeric_to_string(in_list):
    return(' ; '.join([' '.join([str(c) for c in lst]) for lst in in_list]))

def join_nested(left, right, on):
    left['key'] = left[on].apply(convert_nested_numeric_to_string)
    right['key'] = right[on].apply(convert_nested_numeric_to_string)
    out = pd.merge(left.drop(columns=[on]), right, on='key', how='left')
    out = out.drop(columns=['key'])
    return(out)
