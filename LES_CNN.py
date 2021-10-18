# Solve 2D turbulence by Fourier-Fourier pseudo-spectral method
# Navier-Stokes equation is in the vorticity-stream function form

import sys
import pycuda
import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
from skcuda import cufft as cf

import numpy as np
import math
import scipy
from scipy import sparse
from scipy.sparse.linalg import inv
import statistics

import h5py
from scipy.io import loadmat,savemat
import time as runtime


import os
import tensorflow as tf
from keras import layers

from keras.layers import Input, Convolution2D, Convolution1D, MaxPooling2D, Dense, Dropout, \
                          Flatten, concatenate, Activation, Reshape, \
                          UpSampling2D,ZeroPadding2D
from keras.layers import Dense
from keras import Sequential
import h5py
import keras

from tensorflow.compat.v1.keras.backend import set_session

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, dot, Input
import tensorflow.keras.backend as K


from scipy.io import loadmat,savemat

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.3)
tf.compat.v1.disable_eager_execution()
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

ker = SourceModule("""
#include <pycuda-complex.hpp>
#include "cuComplex.h"
#include <cufft.h>
#if __CUDA_ARCH__ < 600
__device__ double atomicAdd(double *address, double val){
	unsigned long long int* address_as_ull = (unsigned long long int*) address;
	unsigned long long int old = *address_as_ull;
	unsigned long long int assumed;
	do{
		assumed = old;
		old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
		// Note: uses integer comparision to avoid hang in case of NaN (since NaN!=NaN)
		} while(assumed != old);
		return __longlong_as_double(old);
}
#endif

const int NX = 128;
const int NX2 = 16384; // NX^2
const int NNX = 2; // Filter size
__device__ double dt = 1e-4;
__device__ double nu = 5e-5;
__device__ double alpha = 0.1;

__global__ void initialization_kernel(cufftDoubleComplex *u0, cufftDoubleComplex *v0, cufftDoubleComplex *w0, double *kx,\
     cufftDoubleComplex *psi)
{
    unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int j = blockIdx.y;

    v0[i*NX+j].x = -kx[j]*psi[j+i*NX].y;
    v0[i*NX+j].y = kx[j]*psi[j+i*NX].x;

    u0[i*NX+j].x = kx[i]*psi[j+i*NX].y;
    u0[i*NX+j].y = -kx[i]*psi[j+i*NX].x;

    w0[i*NX+j].x = -(kx[j]*kx[j] + kx[i]*kx[i])*psi[j+i*NX].x;
    w0[i*NX+j].y = -(kx[j]*kx[j] + kx[i]*kx[i])*psi[j+i*NX].y;
}

__global__ void iniW1_kernel(cufftDoubleComplex *w0, double *kx,\
     cufftDoubleComplex *psi)
{
    unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int j = blockIdx.y;

    w0[i*NX+j].x = -(kx[j]*kx[j] + kx[i]*kx[i])*psi[j+i*NX].x;
    w0[i*NX+j].y = -(kx[j]*kx[j] + kx[i]*kx[i])*psi[j+i*NX].y;
}

__global__ void UV_kernel(cufftDoubleComplex *u0, cufftDoubleComplex *v0, double *kx,\
     cufftDoubleComplex *psi)
{
    unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int j = blockIdx.y;

    v0[i*NX+j].x = -kx[j]*psi[j+i*NX].y;
    v0[i*NX+j].y = kx[j]*psi[j+i*NX].x;

    u0[i*NX+j].x = kx[i]*psi[j+i*NX].y;
    u0[i*NX+j].y = -kx[i]*psi[j+i*NX].x;
}

__global__ void convection2_kernel(cufftDoubleComplex *u, cufftDoubleComplex *w, \
    cufftDoubleComplex *convec)
{
    unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int j = blockIdx.y;

    convec[i*NX+j].x = (u[i*NX+j].x / NX2) * (w[i*NX+j].x / NX2);
    convec[i*NX+j].y = 0.0;//u[i*NX+j].y * w[i*NX+j].y / NX2 / NX2;
}

__global__ void diffusion_kernel(cufftDoubleComplex *diffu, double *kx, cufftDoubleComplex *w1)
{
    unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int j = blockIdx.y;
    diffu[i*NX+j].x = -(kx[j]*kx[j] + kx[i]*kx[i])*w1[j+i*NX].x;
    diffu[i*NX+j].y = -(kx[j]*kx[j] + kx[i]*kx[i])*w1[j+i*NX].y;
}

__global__ void convection3_kernel(cufftDoubleComplex *conu1, cufftDoubleComplex *conv1,\
    cufftDoubleComplex *conu0, cufftDoubleComplex *conv0, cufftDoubleComplex *convN, double *kx, cufftDoubleComplex *convec)
{
    unsigned int j = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int k = blockIdx.y;

    convec[j*NX+k].x = 0.5*dt*(kx[k]*(1.5*conu1[j*NX+k].y - 0.5*conu0[j*NX+k].y) + kx[j]*(1.5*conv1[j*NX+k].y - 0.5*conv0[j*NX+k].y)\
        + convN[j*NX+k].x);
    convec[j*NX+k].y = 0.5*dt*(kx[k]*(-1.5*conu1[j*NX+k].x + 0.5*conu0[j*NX+k].x) + kx[j]*(-1.5*conv1[j*NX+k].x + 0.5*conv0[j*NX+k].x)\
        + convN[j*NX+k].y);
}

__global__ void convection4_kernel(cufftDoubleComplex *wx, cufftDoubleComplex *wy, double *kx,\
     cufftDoubleComplex *w)
{
    unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int j = blockIdx.y;

    wx[i*NX+j].x = -kx[j]*w[j+i*NX].y;
    wx[i*NX+j].y = kx[j]*w[j+i*NX].x;

    wy[i*NX+j].x = -kx[i]*w[j+i*NX].y;
    wy[i*NX+j].y = kx[i]*w[j+i*NX].x;
}

__global__ void convection5_kernel(cufftDoubleComplex *u1,cufftDoubleComplex *v1,cufftDoubleComplex *u0,cufftDoubleComplex *v0, \
    cufftDoubleComplex *w1x,cufftDoubleComplex *w1y,cufftDoubleComplex *w0x,cufftDoubleComplex *w0y,\
        cufftDoubleComplex *convec)
{
    unsigned int j = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int k = blockIdx.y;

    convec[j*NX+k].x = (-1.5*((u1[j*NX+k].x/NX2)*(w1x[j*NX+k].x/NX2) + (v1[j*NX+k].x/NX2)*(w1y[j*NX+k].x/NX2))\
        +0.5*((u0[j*NX+k].x/NX2)*(w0x[j*NX+k].x/NX2) + (v0[j*NX+k].x/NX2)*(w0y[j*NX+k].x/NX2)));
}

__global__ void convection6_kernel(cufftDoubleComplex *u1,cufftDoubleComplex *v1, cufftDoubleComplex *w1x,cufftDoubleComplex *w1y, cufftDoubleComplex *convec)
{
    unsigned int j = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int k = blockIdx.y;

    convec[j*NX+k].x = (u1[j*NX+k].x/NX2)*(w1x[j*NX+k].x/NX2) + (v1[j*NX+k].x/NX2)*(w1y[j*NX+k].x/NX2);
}

__global__ void RHS_kernel(cufftDoubleComplex *convec,cufftDoubleComplex *diffu,cufftDoubleComplex *w1,\
    cufftDoubleComplex *RHS)
{
    unsigned int j = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int k = blockIdx.y;

    RHS[j*NX+k].x = convec[j*NX+k].x + 0.5*dt*nu*diffu[j*NX+k].x + w1[j*NX+k].x;
    RHS[j*NX+k].y = convec[j*NX+k].y + 0.5*dt*nu*diffu[j*NX+k].y + w1[j*NX+k].y;
}

__global__ void LHS_kernel(double *kx, cufftDoubleComplex *RHS, cufftDoubleComplex *w)
{
    unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int j = blockIdx.y;

    w[i*NX+j].x = RHS[i*NX+j].x / (1.0+dt*alpha+0.5*dt*nu*((kx[j]*kx[j] + kx[i]*kx[i])));
    w[i*NX+j].y = RHS[i*NX+j].y / (1.0+dt*alpha+0.5*dt*nu*((kx[j]*kx[j] + kx[i]*kx[i])));

}
__global__ void psiTemp_kernel(cufftDoubleComplex *w, double *kx, cufftDoubleComplex *psiTemp)
{
    unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int j = blockIdx.y;

    if (j==0 && i==0){
        psiTemp[i*NX+j].x = 0.0;
        psiTemp[i*NX+j].y = 0.0;
    }
    else{
        psiTemp[i*NX+j].x =  - w[i*NX+j].x / (kx[j]*kx[j] + kx[i]*kx[i]);
        psiTemp[i*NX+j].y =  - w[i*NX+j].y / (kx[j]*kx[j] + kx[i]*kx[i]);
    }
}

__global__ void update_kernel(cufftDoubleComplex *psiPrevious_hat_gpu,cufftDoubleComplex *psiCurrent_hat_gpu,\
        cufftDoubleComplex *psiTemp_gpu)
    {
        unsigned int j = threadIdx.x + blockIdx.x*blockDim.x;
        unsigned int k = blockIdx.y;
        psiPrevious_hat_gpu[j*NX+k].x = psiCurrent_hat_gpu[j*NX+k].x;
        psiPrevious_hat_gpu[j*NX+k].y = psiCurrent_hat_gpu[j*NX+k].y;
        psiCurrent_hat_gpu[j*NX+k].x  = psiTemp_gpu[j*NX+k].x;
        psiCurrent_hat_gpu[j*NX+k].y  = psiTemp_gpu[j*NX+k].y;
    }

__global__ void updateu_kernel(cufftDoubleComplex *u0_hat_gpu, cufftDoubleComplex *v0_hat_gpu,\
        cufftDoubleComplex *u1_hat_gpu,cufftDoubleComplex *v1_hat_gpu)
    {
        unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
        unsigned int j = blockIdx.y;
        u0_hat_gpu[i*NX+j].x = u1_hat_gpu[i*NX+j].x;
        u0_hat_gpu[i*NX+j].y = u1_hat_gpu[i*NX+j].y;
        v0_hat_gpu[i*NX+j].x = v1_hat_gpu[i*NX+j].x;
        v0_hat_gpu[i*NX+j].y = v1_hat_gpu[i*NX+j].y;
    }

__global__ void updatew_kernel(cufftDoubleComplex *w0_hat_gpu, cufftDoubleComplex *w1_hat_gpu)
    {
        unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
        unsigned int j = blockIdx.y;
        w0_hat_gpu[i*NX+j].x = w1_hat_gpu[i*NX+j].x;
        w0_hat_gpu[i*NX+j].y = w1_hat_gpu[i*NX+j].y;
    }

__global__ void DDP_kernel(cufftDoubleComplex *RHS, cufftDoubleComplex *pred)
{
    unsigned int j = threadIdx.y + blockIdx.y*blockDim.y;
    unsigned int k = blockIdx.z;

    RHS[j*NX+k].x = RHS[j*NX+k].x  + 1*dt*pred[j*NX+k].x;
    RHS[j*NX+k].y = RHS[j*NX+k].y  + 1*dt*pred[j*NX+k].y;
}

__global__ void spectralFilter_2D_kernel(cufftDoubleComplex *data, cufftDoubleComplex *data_F)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;

    int HNNX = NNX/2;
    if (i<=HNNX && j<=HNNX){
        data_F[i*NNX+j].x = data[i*NX+j].x;
        data_F[i*NNX+j].y = data[i*NX+j].y;
    }
    else if (i>(NX-HNNX) && j<=HNNX){
        data_F[(NNX+i-NX)*NNX+j].x = data[i*NX+j].x;
        data_F[(NNX+i-NX)*NNX+j].y = data[i*NX+j].y;
    }
    else if (j>(NX-HNNX) && i<=HNNX){
        data_F[i*NNX+(NNX+j-NX)].x = data[i*NX+j].x;
        data_F[i*NNX+(NNX+j-NX)].y = data[i*NX+j].y;
    }
    else if (j>(NX-HNNX) && i>(NX-HNNX)){
        data_F[(NNX+i-NX)*NNX+(NNX+j-NX)].x = data[i*NX+j].x;
        data_F[(NNX+i-NX)*NNX+(NNX+j-NX)].y = data[i*NX+j].y;
    }
}

__global__ void addConvecC_kernel(cufftDoubleComplex *conu1, cufftDoubleComplex *conv1, double *kx, cufftDoubleComplex *convecC)
{
    unsigned int j = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int k = blockIdx.y;

    convecC[j*NX+k].x = -(kx[k]*conu1[j*NX+k].y  + kx[j]*conv1[j*NX+k].y);
    convecC[j*NX+k].y = (kx[k]*conu1[j*NX+k].x  + kx[j]*conv1[j*NX+k].x);
}

__global__ void GaussianFilter_2D_kernel(cufftDoubleComplex *u, double *Gk)
{
    unsigned int j = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int k = blockIdx.y;

    u[j*NX+k].x = Gk[j*NX+k]*u[j*NX+k].x;
    u[j*NX+k].y = Gk[j*NX+k]*u[j*NX+k].y;
}


""")
initialization = ker.get_function("initialization_kernel")
iniW1 = ker.get_function("iniW1_kernel")
UV_ker = ker.get_function("UV_kernel")
convection2 = ker.get_function("convection2_kernel")
convection3 = ker.get_function("convection3_kernel")
convection4 = ker.get_function("convection4_kernel")
convection5 = ker.get_function("convection5_kernel")
convection6 = ker.get_function("convection6_kernel")
diffusion = ker.get_function("diffusion_kernel")
RHS_ker = ker.get_function("RHS_kernel")
LHS_ker = ker.get_function("LHS_kernel")
psiTemp_ker = ker.get_function("psiTemp_kernel")
update = ker.get_function("update_kernel")
updateu = ker.get_function("updateu_kernel")
updatew = ker.get_function("updatew_kernel")
DDP_ker = ker.get_function("DDP_kernel")
spectralFilter = ker.get_function("spectralFilter_2D_kernel")
GaussianFilter = ker.get_function("GaussianFilter_2D_kernel")
addConvecC = ker.get_function("addConvecC_kernel")

dataset = 1

NSAVE = 100
NNSAVE = 1000
maxit = NSAVE*NNSAVE+1

dt    = 5e-4
nu    = 1e-5
rho   = 1
Re    = 1.0/nu

Lx    = 2*math.pi
NX    = 128
HNX   = 128


# Neural network setup

model = keras.models.load_model('./Models/CNN_TL_from_Re_1k_to_Re_100k_per_train_10_layers_2')

#--> Computational variables
TPB = 1
NB  = NX//TPB
HNB = NB//2
HNB = HNX
TPBx = 1

## No mirror symmetry
HNB = NB
HNX = NX

dx    = Lx/NX
x     = np.linspace(0, Lx-dx, num=NX)
kx    = (2*math.pi/Lx)*np.concatenate((np.arange(0,NX/2+1,dtype=np.float64),np.arange((-NX/2+1),0,dtype=np.float64)))

# Create the tensor product mesh
[Y,X]       = np.meshgrid(x,x)
[Ky,Kx]     = np.meshgrid(kx,kx)
Ksq         = (Kx**2 + Ky**2)
invKsq      = 1/Ksq
invKsq[0,0] = 0

data_Poi = loadmat('../test/iniWor'+dataset+'.mat')
w1 = data_Poi['w1']
w1 = w1.T
w1_hat = np.fft.fft2(w1)
psiCurrent_hat = -invKsq*w1_hat
psiPrevious_hat = psiCurrent_hat

time = 0.0
slnW = []
print('Re = ', Re)

# Deterministic forcing
n = 4
Xi = 1
Fk = -n*Xi*(np.cos(n*Y))
Fk = np.fft.fft2(Fk)

Force_gpu   = gpuarray.to_gpu(np.reshape(Fk,NX*(NX),order='F'))

plan = cf.cufftPlan2d(NX, NX, cf.CUFFT_Z2Z)

slnU = np.zeros([NX,NNSAVE+1])
slnV = np.zeros([NX,NNSAVE+1])
onePython = np.zeros([NNSAVE+1])
slnWor = np.zeros([NX,NX,NNSAVE+1])
SGS = np.zeros([NX,NX,NNSAVE+1])
Energy = np.zeros([NNSAVE+1])
Enstrophy = np.zeros([NNSAVE+1])
count = 0

# DDP input
input_data=np.zeros([1, NX, NX, 2])

start_time = runtime.time()
for it in range(maxit):
    if it == 0:
        # Sampling matrix as for Maulik's model
        sampling_matrix = np.zeros([20,NX,NX],dtype=np.float64)
        sampling_matrix_1 = np.zeros([NX*NX+1,20],dtype=np.float64)
        # On the first iteration
        w1_hat = np.zeros([NX,NX],dtype=np.complex128)
        u1_hat = np.zeros([NX,NX],dtype=np.complex128)
        v1_hat = np.zeros([NX,NX],dtype=np.complex128)
        w0_hat = np.zeros([NX,NX],dtype=np.complex128)
        u0_hat = np.zeros([NX,NX],dtype=np.complex128)
        v0_hat = np.zeros([NX,NX],dtype=np.complex128)
        diffu_hat = np.zeros([NX,NX],dtype=np.complex128)
        diffu = np.zeros([NX,NX],dtype=np.complex128)
        convec = np.zeros([NX,NX],dtype=np.complex128)
        convec_hat = np.zeros([NX,NX],dtype=np.complex128)
        RHS = np.zeros([NX,NX],dtype=np.complex128)
        psiTemp = np.zeros([NX,NX],dtype=np.complex128)

        u1 = np.zeros([NX,NX],dtype=np.complex128)
        u0 = np.zeros([NX,NX],dtype=np.complex128)
        v1 = np.zeros([NX,NX],dtype=np.complex128)
        v0 = np.zeros([NX,NX],dtype=np.complex128)

        w1 = np.zeros([NX,NX],dtype=np.complex128)
        w0 = np.zeros([NX,NX],dtype=np.complex128)

        w1x_hat = np.zeros([NX,NX],dtype=np.complex128)
        w1y_hat = np.zeros([NX,NX],dtype=np.complex128)
        w0x_hat = np.zeros([NX,NX],dtype=np.complex128)
        w0y_hat = np.zeros([NX,NX],dtype=np.complex128)
        w1x = np.zeros([NX,NX],dtype=np.complex128)
        w1y = np.zeros([NX,NX],dtype=np.complex128)
        w0x = np.zeros([NX,NX],dtype=np.complex128)
        w0y = np.zeros([NX,NX],dtype=np.complex128)

        psi1 = np.zeros([NX,NX],dtype=np.complex128)

        conu1 = np.zeros([NX,NX],dtype=np.complex128)
        conv1 = np.zeros([NX,NX],dtype=np.complex128)
        conu0 = np.zeros([NX,NX],dtype=np.complex128)
        conv0 = np.zeros([NX,NX],dtype=np.complex128)
        conu1_hat = np.zeros([NX,NX],dtype=np.complex128)
        conv1_hat = np.zeros([NX,NX],dtype=np.complex128)
        conu0_hat = np.zeros([NX,NX],dtype=np.complex128)
        conv0_hat = np.zeros([NX,NX],dtype=np.complex128)

        convN_hat = np.zeros([NX,NX],dtype=np.complex128)

        w1_hat_gpu = gpuarray.to_gpu(np.reshape(w1_hat,NX*(NX),order='F'))
        u1_hat_gpu = gpuarray.to_gpu(np.reshape(u1_hat,NX*(NX),order='F'))
        v1_hat_gpu = gpuarray.to_gpu(np.reshape(v1_hat,NX*(NX),order='F'))
        w0_hat_gpu = gpuarray.to_gpu(np.reshape(w0_hat,NX*(NX),order='F'))
        u0_hat_gpu = gpuarray.to_gpu(np.reshape(u0_hat,NX*(NX),order='F'))
        v0_hat_gpu = gpuarray.to_gpu(np.reshape(v0_hat,NX*(NX),order='F'))
        w1_gpu = gpuarray.to_gpu(np.reshape(w1,NX*(NX),order='F'))
        w0_gpu = gpuarray.to_gpu(np.reshape(w0,NX*(NX),order='F'))
        conu1_gpu = gpuarray.to_gpu(np.reshape(conu1,NX*(NX),order='F'))
        conv1_gpu = gpuarray.to_gpu(np.reshape(conv1,NX*(NX),order='F'))
        conu0_gpu = gpuarray.to_gpu(np.reshape(conu0,NX*(NX),order='F'))
        conv0_gpu = gpuarray.to_gpu(np.reshape(conv0,NX*(NX),order='F'))
        conu1_hat_gpu = gpuarray.to_gpu(np.reshape(conu1_hat,NX*(NX),order='F'))
        conv1_hat_gpu = gpuarray.to_gpu(np.reshape(conv1_hat,NX*(NX),order='F'))
        conu0_hat_gpu = gpuarray.to_gpu(np.reshape(conu0_hat,NX*(NX),order='F'))
        conv0_hat_gpu = gpuarray.to_gpu(np.reshape(conv0_hat,NX*(NX),order='F'))
        convecN_hat_gpu = gpuarray.to_gpu(np.reshape(convN_hat,NX*(NX),order='F'))

        w1x_hat_gpu = gpuarray.to_gpu(np.reshape(w1x_hat,NX*(NX),order='F'))
        w1y_hat_gpu = gpuarray.to_gpu(np.reshape(w1y_hat,NX*(NX),order='F'))
        w0x_hat_gpu = gpuarray.to_gpu(np.reshape(w0x_hat,NX*(NX),order='F'))
        w0y_hat_gpu = gpuarray.to_gpu(np.reshape(w0y_hat,NX*(NX),order='F'))
        w1x_gpu = gpuarray.to_gpu(np.reshape(w1x,NX*(NX),order='F'))
        w1y_gpu = gpuarray.to_gpu(np.reshape(w1y,NX*(NX),order='F'))
        w0x_gpu = gpuarray.to_gpu(np.reshape(w0x,NX*(NX),order='F'))
        w0y_gpu = gpuarray.to_gpu(np.reshape(w0y,NX*(NX),order='F'))


        diffu_gpu = gpuarray.to_gpu(np.reshape(diffu,NX*(NX),order='F'))
        diffu_hat_gpu = gpuarray.to_gpu(np.reshape(diffu_hat,NX*(NX),order='F'))
        convec_gpu = gpuarray.to_gpu(np.reshape(convec,NX*(NX),order='F'))
        convec_hat_gpu = gpuarray.to_gpu(np.reshape(convec_hat,NX*(NX),order='F'))
        RHS_gpu = gpuarray.to_gpu(np.reshape(RHS,NX*(NX),order='F'))
        psiTemp_gpu = gpuarray.to_gpu(np.reshape(psiTemp,NX*(NX),order='F'))

        u1_gpu = gpuarray.to_gpu(np.reshape(u1,NX*(NX),order='F'))
        v1_gpu = gpuarray.to_gpu(np.reshape(v1,NX*(NX),order='F'))
        u0_gpu = gpuarray.to_gpu(np.reshape(u0,NX*(NX),order='F'))
        v0_gpu = gpuarray.to_gpu(np.reshape(v0,NX*(NX),order='F'))
        psi1_gpu = gpuarray.to_gpu(np.reshape(psi1,NX*(NX),order='F'))

        kx      = kx.astype(np.float64)

        psiPrevious_hat_gpu = gpuarray.to_gpu(np.reshape(psiPrevious_hat,NX*(NX),order='F'))
        psiCurrent_hat_gpu  = gpuarray.to_gpu(np.reshape(psiCurrent_hat,NX*(NX),order='F'))
        kx_gpu              = gpuarray.to_gpu(kx)

        initialization(u0_hat_gpu, v0_hat_gpu, w0_hat_gpu, kx_gpu, psiPrevious_hat_gpu, block=(TPB,1,1), grid=(HNB,HNX,1))

        iniW1(w1_hat_gpu, kx_gpu, psiCurrent_hat_gpu, block=(TPB,1,1), grid=(HNB,HNX,1))

        runtime2 = runtime.time()

    else:
        updateu(u0_hat_gpu, v0_hat_gpu, u1_hat_gpu, v1_hat_gpu, block=(TPB,1,1), grid=(NB,NX,1))

    UV_ker(u1_hat_gpu, v1_hat_gpu, kx_gpu, psiCurrent_hat_gpu,
        block=(TPB,1,1), grid=(HNB,HNX,1))


    diffusion(diffu_hat_gpu, kx_gpu, w1_hat_gpu, block=(TPB,1,1), grid=(HNB,HNX,1))

    # Conservative convection form

    cf.cufftExecZ2Z(plan, int(u1_hat_gpu.gpudata), int(u1_gpu.gpudata), cf.CUFFT_INVERSE)
    cf.cufftExecZ2Z(plan, int(v1_hat_gpu.gpudata), int(v1_gpu.gpudata), cf.CUFFT_INVERSE)
    cf.cufftExecZ2Z(plan, int(u0_hat_gpu.gpudata), int(u0_gpu.gpudata), cf.CUFFT_INVERSE)
    cf.cufftExecZ2Z(plan, int(v0_hat_gpu.gpudata), int(v0_gpu.gpudata), cf.CUFFT_INVERSE)
    cf.cufftExecZ2Z(plan, int(w1_hat_gpu.gpudata), int(w1_gpu.gpudata), cf.CUFFT_INVERSE)
    cf.cufftExecZ2Z(plan, int(w0_hat_gpu.gpudata), int(w0_gpu.gpudata), cf.CUFFT_INVERSE)

    convection2(u1_gpu, w1_gpu, conu1_gpu, block=(TPB,1,1), grid=(HNB,HNX,1))
    convection2(v1_gpu, w1_gpu, conv1_gpu, block=(TPB,1,1), grid=(HNB,HNX,1))
    convection2(u0_gpu, w0_gpu, conu0_gpu, block=(TPB,1,1), grid=(HNB,HNX,1))
    convection2(v0_gpu, w0_gpu, conv0_gpu, block=(TPB,1,1), grid=(HNB,HNX,1))

    cf.cufftExecZ2Z(plan, int(conu1_gpu.gpudata), int(conu1_hat_gpu.gpudata), cf.CUFFT_FORWARD)
    cf.cufftExecZ2Z(plan, int(conv1_gpu.gpudata), int(conv1_hat_gpu.gpudata), cf.CUFFT_FORWARD)
    cf.cufftExecZ2Z(plan, int(conu0_gpu.gpudata), int(conu0_hat_gpu.gpudata), cf.CUFFT_FORWARD)
    cf.cufftExecZ2Z(plan, int(conv0_gpu.gpudata), int(conv0_hat_gpu.gpudata), cf.CUFFT_FORWARD)

    # Non-conservative convection form
    convection4(w0x_hat_gpu, w0y_hat_gpu, kx_gpu, w0_hat_gpu, block=(TPB,1,1), grid=(HNB,HNX,1))

    convection4(w1x_hat_gpu, w1y_hat_gpu, kx_gpu, w1_hat_gpu, block=(TPB,1,1), grid=(HNB,HNX,1))

    cf.cufftExecZ2Z(plan, int(w1x_hat_gpu.gpudata), int(w1x_gpu.gpudata), cf.CUFFT_INVERSE)
    cf.cufftExecZ2Z(plan, int(w1y_hat_gpu.gpudata), int(w1y_gpu.gpudata), cf.CUFFT_INVERSE)
    cf.cufftExecZ2Z(plan, int(w0x_hat_gpu.gpudata), int(w0x_gpu.gpudata), cf.CUFFT_INVERSE)
    cf.cufftExecZ2Z(plan, int(w0y_hat_gpu.gpudata), int(w0y_gpu.gpudata), cf.CUFFT_INVERSE)

    convection5(u1_gpu, v1_gpu, u0_gpu, v0_gpu, w1x_gpu, w1y_gpu, w0x_gpu, w0y_gpu, convec_gpu, block=(TPB,1,1), grid=(HNB,HNX,1))

    cf.cufftExecZ2Z(plan, int(convec_gpu.gpudata), int(convecN_hat_gpu.gpudata), cf.CUFFT_FORWARD)


    # Convection = 0.5*(convec + convecN)
    convection3(conu1_hat_gpu, conv1_hat_gpu, conu0_hat_gpu, conv0_hat_gpu, convecN_hat_gpu, kx_gpu, convec_hat_gpu, block=(TPB,1,1), grid=(NB,NX,1))


    RHS_ker(convec_hat_gpu, diffu_hat_gpu, w1_hat_gpu, RHS_gpu, block=(TPB,1,1), grid=(NB,NX,1))

    # Step n
    cf.cufftExecZ2Z(plan, int(w1_hat_gpu.gpudata), int(w1_gpu.gpudata), cf.CUFFT_INVERSE)
    cf.cufftExecZ2Z(plan, int(psiCurrent_hat_gpu.gpudata), int(psi1_gpu.gpudata), cf.CUFFT_INVERSE)
    psiDDP = -1*np.real(np.reshape(psi1_gpu.get(),[NX,NX],order='F'))/(NX*NX)# Get the real stream function
    wDDP = np.real(np.reshape(w1_gpu.get(),[NX,NX],order='F'))/(NX*NX)

    #psiDDP = -psiDDP # Get the real stream function
    input_data[0,:,:,0] = (psiDDP.T - psiDDP.mean())/psiDDP.std()
    input_data[0,:,:,1] = (wDDP.T - wDDP.mean())/wDDP.std()


    prediction = model.predict(input_data)

    SDEV_O     = 2.1592
    MEAN_O     = 0.0
    prediction1 = np.reshape(prediction,[NX,NX]) * SDEV_O + MEAN_O

    prediction = -1*prediction1.T
    prediction = np.complex128(prediction)

    pred_hat = np.fft.fft2(prediction)

    pred_gpu   = gpuarray.to_gpu(np.reshape(pred_hat,NX*(NX),order='F'))

    # Couple SGS to RHS
    DDP_ker(RHS_gpu, pred_gpu, block=(1,TPB,1), grid=(1,NB,NX))

    # Deterministic forcing
    DDP_ker(RHS_gpu, Force_gpu, block=(1,TPB,1), grid=(1,NB,NX))
    # End forcing

    updatew(w0_hat_gpu, w1_hat_gpu, block=(TPB,1,1), grid=(NB,NX,1))

    LHS_ker(kx_gpu, RHS_gpu, w1_hat_gpu, block=(TPB,1,1), grid=(HNB,HNX,1))

    psiTemp_ker(w1_hat_gpu, kx_gpu, psiTemp_gpu, block=(TPB,1,1), grid=(NB,NX,1))

    update(psiPrevious_hat_gpu, psiCurrent_hat_gpu, psiTemp_gpu, block=(TPB,1,1), grid=(NB,NX,1))

    time_LHS = runtime.time()-runtime2
    runtim2 = runtime.time()

    time = time + dt


    if np.mod(it, NSAVE) == 0:
        cf.cufftExecZ2Z(plan, int(w0_hat_gpu.gpudata), int(u1_gpu.gpudata), cf.CUFFT_INVERSE)
        u = np.real(np.reshape(u1_gpu.get(),[NX,NX],order='F'))/(NX*NX)
        slnWor[:,:,count] = u

        cf.cufftExecZ2Z(plan, int(psiPrevious_hat_gpu.gpudata), int(u1_gpu.gpudata), cf.CUFFT_INVERSE)
        psi = -np.real(np.reshape(u1_gpu.get(),[NX,NX],order='F'))/(NX*NX)

        ener = u*psi
        enst = u*u

        Energy[count] = np.sum(ener)
        Enstrophy[count] = np.sum(enst)

        tempW = np.max(np.squeeze(u))

        slnW.append(tempW)

        print(it)
        print(tempW)
        print("--- %s seconds ---" % (runtime.time() - runtime2))
        runtime2 = runtime.time()

        cf.cufftExecZ2Z(plan, int(u1_hat_gpu.gpudata), int(u1_gpu.gpudata), cf.CUFFT_INVERSE)
        cf.cufftExecZ2Z(plan, int(v1_hat_gpu.gpudata), int(v1_gpu.gpudata), cf.CUFFT_INVERSE)

        u = np.real(np.reshape(u1_gpu.get(),[NX,NX],order='F'))/(NX*NX)
        v = np.real(np.reshape(v1_gpu.get(),[NX,NX],order='F'))/(NX*NX)

        slnU[:,count] = u[:,0]
        slnV[:,count] = v[:,0]

        SGS[:,:,count] = prediction1.T

        count = count + 1



print("--- %s seconds ---(end iteration)" % (runtime.time() - start_time))

psiCurrent_hat = psiCurrent_hat_gpu.get()
psiPrevious_hat = psiPrevious_hat_gpu.get()

psi = psiCurrent_hat
savemat('data_end_LDC_Poi_gpu_Kraichnan.mat', dict([('psiCurrent_hat', psiCurrent_hat), ('psiPrevious_hat', psiPrevious_hat), ('time', time)
, ('slnW', slnW)]))
savemat('Energy_Re100k_'+dataset+'_NSGS.mat',dict([('Energy',Energy),('Enstrophy',Enstrophy)]))
print("--- %s seconds ---" % (runtime.time() - start_time))


savemat('w_LES_Re_100k_'+dataset+'_NSGS.mat',dict([('slnWorCNN', slnWor),('slnSGS', SGS), ('slnW', slnW)]))

print(time)
