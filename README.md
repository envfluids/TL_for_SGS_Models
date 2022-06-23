# TL_for_SGS_Models 


## Introduction

This repository includes the codes to produce datasets and implement the training, LES, and analyses in the accompanying paper *Explaining the physics of transfer learning a data-driven subgrid-scale closure to a different turbulent flow* [https://arxiv.org/abs/2206.03198](https://arxiv.org/abs/2206.03198). The following links to the datasets that can be used for the training of networks, [https://zenodo.org/record/6621142](https://zenodo.org/record/6621142).

## Requirements
- python 3.8
	- [scipy](https://pypi.org/project/scipy/)
	- [numpy](https://pypi.org/project/numpy/)
- [TensorFlow 2](https://www.tensorflow.org/install)
- [Keras 2.4.3](https://pypi.org/project/Keras/)
- Matlab

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
@article{subel2022explaining,
  title={Explaining the physics of transfer learning a data-driven subgrid-scale closure to a different turbulent flow},
  author={Subel, Adam and Guan, Yifei and Chattopadhyay, Ashesh and Hassanzadeh, Pedram},
  journal={arXiv preprint arXiv:2206.03198},
  year={2022}
}


