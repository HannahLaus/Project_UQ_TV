import matplotlib.pyplot as plt
import os
import numpy as np
import statsmodels.api as sm
import pylops
import math
import skimage
from skimage.color import rgb2gray
import matplotlib.colors as colors
import scipy.io
import cupy as cp

"""
CALCULATION OF THE PRECISION MATRIX M USING THE NODEWISE LASSO.

This script applies the nodewise LASSO to a given mask matrix. 
It uses the package cupy for GPU acceleration/usage. 
If you don't have a GPU available you can change cp to np to always use numpy.
"""
cp.cuda.Device(device=2).use()
matrixtype = 'mat'
""" Choose the correct matrix type here:"""
if matrixtype == 'mat':
    mask = scipy.io.loadmat("radial_full_mask_131_70per.dat")  # exchange for the right mask
    mask = mask['k_space_mask']
elif matrixtype == 'dat':
    mask = np.load("radial_full_mask_131_70per.dat", allow_pickle=True)  # exchange for the right mask
else:
    raise Exception

mask = cp.array(mask)
idces = cp.nonzero(mask.flatten())[0]
ny, nx = mask.shape
print('Length of image', nx)
P = pylops.Restriction(nx*ny, idces, axis=0, dtype=np.complex128)
nxysub = P.shape[0]
print('Number of undersampled elements', nxysub)
F = pylops.signalprocessing.FFT2D(dims=(ny, nx), norm="none", fftshift_after=True, ifftshift_before=True)

eye = cp.eye(nx*nx)


"""Choose regularization parameter and lambda"""
regulparameter = 1/(np.sqrt(12*np.log(nx*ny))/np.sqrt(nxysub))
print(regulparameter)
lamda = [0.0035*(regulparameter)]

"""Initalize tao and C"""
matrixC = cp.zeros((nx*nx,nx*nx), dtype=cp.complex64)
tao = cp.zeros(nx*ny, dtype=cp.complex64)

"""Compute the nodewise LASSO per node i with FISTA"""
for i in range(nx*ny):
    print(i)
    diagonal = cp.ones(nx*ny)
    diagonal[i] = 0
    eye_op = pylops.Diagonal(diagonal)
    datavector = P*F*eye[:, i]
    xinv = pylops.optimization.sparsity.fista(
        1/np.sqrt(nxysub)*P * F * eye_op,
        1/np.sqrt(nxysub)*datavector,
        niter=1000,
        eps=lamda[0],
        tol=1e-25,
    )[0]
    tao[i] = (1/nxysub)*cp.vdot((datavector-P*F*eye_op*xinv),datavector)
    tao[i] = np.conjugate(tao[i])
    xinv = -xinv
    xinv[i] = 1
    matrixC[i, :] = xinv


del eye
del datavector
del diagonal
"Compute M"
M = (1/tao)*matrixC
print("M", M)

del matrixC
del tao
idces = cp.asnumpy(idces)
P = pylops.Restriction(nx*ny, idces, axis=0, dtype=np.complex128)
"Compute Covariance Matrix"
true_cov = ((1/nxysub)*(P*F).H *P * F) * np.eye(nx*ny)
print(true_cov[0])
"Save M"
cp.asnumpy(M).dump("M_radial.dat", protocol=4)
"Check multiplication M*samplecov"
print('Approximate identity matrix', np.matmul(cp.asnumpy(M), true_cov[:,0]))
print(np.max(np.abs(np.real(np.matmul(cp.asnumpy(M), true_cov[0])-np.eye(nx*ny)[0]))))
