# TL_for_SGS_Models 


## Introduction

This is the introduction. We plan to
<ul>
<li>Do the first item,</li>
<li>Test the second item,</li>
<li>Investigate the third item,</li>
<li>Code the fourth item.</li>
</ul>

## Requirements
- python 3.8
	- [scipy](https://pypi.org/project/scipy/)
	- [numpy](https://pypi.org/project/numpy/)
- [TensorFlow 2](https://www.tensorflow.org/install)
- [Keras 2.4.3](https://pypi.org/project/Keras/)

## Codes
### Train BNN (Train_BNN.py)
Code takes in training and validation data sets to train a new BNN from a random initialization. This outputs the trained model as well as the predictions of the trained model on a test set of data.

### Transfer Learning (DDP_CNN_TL_Single_Layers.py and DDP_CNN_TL_Two_Layers.py)
Code takes in training and validation data as well as a trained BNN to perform transfer learning. The code for two layers is easily modified to select any combination of any number of layers.

### Network Post Processing (Extract_Activations.py, Extract_Activations_Linear.py, and Extract_Weights.py)
These codes all take a trained BNN or TLNN and extract out the weights or activation to a .mat format for later analysis. The code Extract_Activations_Linear.py computes the activations, but removes any nonlinearity after the final layer before outputting activations. 

### Coupled LES (LES_CNN.py)
This code is used for the online testing. This code takes in a trained NN and an initial condition to generate data from large eddy simulation.

### Visualizing Kernel Spectra (Plot_Kernel_Spectra.m)
This take the extracted network weights, computes the kernels with the largest changes due to re-training and plots the spectrum of the kernel from both the BNN and TLNN.

## Citation



