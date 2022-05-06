# CDST (Calvin's Data Science Toolbox)

CDST is a collection of data science Python library developed by Calvin Chan at DSAA, Bayer Pharmaceutical. It contains various data science toolsets mostly based on deep learning technique:

- General Scalable Deep Learning Fully Connected Network (DNN)
- Calvin's Scalable Parallel Downsampler (CSPD)
- Ordinal Hyperplane Loss Classifier (OHPL)

The above algorithms are written to deal with positive output data, updates will be made in the future to accomodate real number upon requests.

This package allows users to sample the network architecture based on sampling parameter, the architecture sampling function is included in this package. The architecture sampling parameter is used as hyperparameter and the user can sample the network architecture based on: (1) a given number of neutrons or (2) a given number of model parameters. In the case of using a given number of model parameters, the sample is computed based on Mixed-Integer Nonlinear Programming Model using the GEKKO package. The accuracy/error of the given set of hyperparameter is estimated using k-fold cross validation, the accuracy/error of each of the k-fold is returned for statistical analysis.

All deep learning modules in this package are designed based on the Ray Tune hyperparameter tuning package, user can sample the multi-layer network neuron distribution using the provided architecture sampling function, together with the range of other hyperparameters including: learning rate, batch size, dropout probability. 

Design examples are shown in the "[example](https://github.com/Bayer-Group/cdst/tree/main/example)" folder with detail structure and graphical illustration of each module. Users can follow these examples and adjust accordingly to suit their own use case and to better understand the mechanics behind the package.


## Hyperparameter Tunning

### DNN

* Use __custom sampling function__ to describe the hierachical neuron distribution between:
    * total neuron: <img src="https://render.githubusercontent.com/render/math?math=H_{total}">
    * neuron per layer: <img src="https://render.githubusercontent.com/render/math?math=H_{layer}">

      <img src="https://render.githubusercontent.com/render/math?math=H_{total}=15\quad\longrightarrow\quad H_{layer}=\begin{bmatrix}3\\4\\5\\3\end{bmatrix}">
    

### CSPD

* Use __custom sampling function__ to describe the hierachical neuron distribution between:
    * total neuron: <img src="https://render.githubusercontent.com/render/math?math=H_{total}">
    * neuron per subgroup: <img src="https://render.githubusercontent.com/render/math?math=H_{subgroup}">
    * neuron per layer: <img src="https://render.githubusercontent.com/render/math?math=H_{branch}">

      <img src="https://render.githubusercontent.com/render/math?math=H_{total}=15\quad\longrightarrow\quad H_{subgroup}=\begin{bmatrix}3\\4\\5\\3\end{bmatrix}\quad \longrightarrow\quad H_{branch}=\begin{bmatrix}[2,1]\\ [2,2]\\ [2,2,1]\\ [1,2] \end{bmatrix}">
    
### OHPL

* Use __custom sampling function__ to describe the hierachical neuron distribution between:
    * total neuron: <img src="https://render.githubusercontent.com/render/math?math=H_{total}">
    * neuron per layer: <img src="https://render.githubusercontent.com/render/math?math=H_{layer}">

      <img src="https://render.githubusercontent.com/render/math?math=H_{total}=15\quad\longrightarrow\quad H_{layer}=\begin{bmatrix}3\\4\\5\\3\end{bmatrix} ">

    
## __Custom Sampling Function__

`split_sampling(num_ele, num_layers=None, n_min=1, n_max=None, n_samples=1, prepend=[], postpend=[], single_sample=False)`

```
   num_ele: Total number of elements to be distributed
   n_min: Minimum number of elements per output dimension
   n_max: Maximum number of elements per output dimension
   num_layers: Number of layers to distribute the element, random dimensions will be given with None given
```

 `parameters_sampling(num_params, num_layers, in_dim, out_dim=1, n_min=1, n_max=None, n_samples=1, include_inout=True, single_sample=False, max_trials=1000)`
 
```
   num_params: Total number of parameters to be distributed
   num_layers: Total number of layers
   n_min:      Minimum number of neurons per layer
   n_max:      Maximum number of neurons per layer
   in_dim:     Number of neurons at the input layer
   out_dim:    Number of neurons at the output layer
   n_samples:  Number of architecture samples to return (maximum number of samples return if there are less than demanded)
   include_inout: Flag indicate whether to include input and output layer neurons with samples
   max_trials: Maximum number of randomized trial for solution sampling if not enough samples found
```

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install CDST.

```bash
pip install git+https://github.com/Bayer-Group/cdst.git
```

## Usage

```python
import cdst

```

## Contributing
For major changes, please open an issue first to discuss what you would like to change. For collaborative development, please initiate developement branch in the git repository and submit for approval prior merging into the master branch.

Please make sure to update tests as appropriate.

## License
[BSD-3-Clause License](https://github.com/bayer-int/cdst/blob/master/LICENSE)

Written by Calvin W.Y. Chan <calvin.chan@bayer.com>, March 2022
(Github: https://github.com/calvinwy, Linkedin: https://www.linkedin.com/in/calchan/)
