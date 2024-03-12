import matplotlib.pyplot as plt
import os
import numpy as np
import statsmodels.api as sm
import pylops
import math
import scipy

"""Compute sample covariance given the mask"""

"Load mask"
sampling = "radial"

if sampling == "radial":
    mask = np.load("radial_mask.dat", allow_pickle=True) #exchange for the right mask
elif sampling == "spiral":
    mask = scipy.io.loadmat("k_space_mask_cart_w_edges_spiral_undersampling_factor_3.mat") #exchange for the right mask
    mask = mask['k_space_mask']
    mask = mask[50:206, 50:206]
else:
    print('Matrixtype not available')
    raise Exception

print('Mask shape', np.shape(mask))

idces = np.nonzero(mask.flatten())[0]
ny, nx = mask.shape
"""Calculate operator A=P*F"""
P = pylops.Restriction(nx*ny, idces, axis=0, dtype=np.complex128)
nxysub = P.shape[0]
F = pylops.signalprocessing.FFT2D(dims=(ny, nx), norm="none", fftshift_after=True, ifftshift_before=True)

"""Compute sample covariance and save it"""
samplecovariance = 1/nxysub*(R*F).H*R*F*np.eye(nx*nx)
samplecovariance.dump("samplecov_radial.dat", protocol=4)

