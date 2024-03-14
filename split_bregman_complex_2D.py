
import matplotlib.pyplot as plt
import os
import numpy as np
import statsmodels.api as sm
import pylops
import math
from phantominator import shepp_logan
import skimage
from skimage.color import rgb2gray
import matplotlib.colors as colors
from skimage.metrics import structural_similarity as ssim


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

"""Input image"""
image = shepp_logan((156,156,5), MR=True, zlims=(-.25,.25))[0][:,:,0]
image = image/np.linalg.norm(image)
ny, nx = image.shape
image_abs = np.abs(image)

"""Choose sampling type"""
sampling = "radial"

"""Choose sigma"""
sigma = 0.1

"""Choose paramters for Split bregman """
mu_factor = 20
lamda = [0.05, 0.05]
niter_out = 1000
niter_in = 50

"""Load mask, samplecovariance matrix and M and calculate diagonal"""
if sampling == "radial":
    samplecovariance = np.load("samplecov_radial.dat",allow_pickle=True)  # exchange here for the right masked
    M = np.load("M_radial.dat", allow_pickle=True)
    Msamplecov = np.matmul(M, samplecovariance)
    diagonal = np.diagonal(np.matmul(np.matmul(M, samplecovariance), np.conjugate(np.transpose(M))))
    mask = np.load("radial_mask.dat", allow_pickle=True)
elif sampling == "spiral":
    samplecovariance = np.load("samplecov_spiraldat", allow_pickle=True)  #exchange here for the right masked
    M = np.load("M_spiral.dat", allow_pickle=True)
    Msamplecov = np.matmul(M, samplecovariance)
    diagonal = np.diagonal(np.matmul(np.matmul(M, samplecovariance), np.conjugate(np.transpose(M))))
    mask = np.load("k_space_mask_spiral_undersampling_factor_3.0.npy")
else:
    print('Matrixtype not available')
    raise Exception

"""Calculate operator A=P*F """
iava = np.nonzero(mask.flatten())[0]


P = pylops.Restriction(nx*ny, iava, axis=0, dtype=np.complex128)
nxysub = P.shape[0]
F = pylops.signalprocessing.FFT2D(dims=(ny, nx), norm="none", fftshift_after=True, ifftshift_before=True)
n_full = sigma/(np.sqrt(2)) * np.random.normal(0, 1, (ny, nx, 2)).view(np.complex128)[:,:,0]
n = P * n_full.ravel()
y = P * ((F * image).ravel() + n_full.ravel())
print('noise rate', np.linalg.norm(n)/np.linalg.norm(P * F * image.ravel()))
yfft = F * image + n_full
ymask = P.mask(F * image+n_full).reshape(nx,ny)


"""Plot model, full kspace data, undersampled k-space data and mask"""
fig, axs = plt.subplots(1, 3, figsize=(14, 5))
im=axs[0].imshow(image_abs, cmap="gray")
axs[0].set_title("Model")
axs[0].axis("tight")
plt.colorbar(im, ax=axs[0])
im = axs[1].imshow(np.abs(yfft),norm=colors.LogNorm(vmin=np.abs(yfft).min(), vmax=np.abs(yfft).max()), cmap="rainbow")
axs[1].set_title("Full data")
axs[1].axis("tight")
plt.colorbar(im, ax=axs[1])
axs[2].imshow(np.abs(ymask), norm=colors.LogNorm(vmin=np.abs(yfft).min(), vmax=np.abs(yfft).max()), cmap="rainbow")
axs[2].set_title("Sampled data")
axs[2].axis("tight")
plt.colorbar(im, ax=axs[2])
plt.tight_layout()
plt.savefig('model_full_data_sampled.pdf')

plt.figure(2)
plt.imshow(mask.astype(int), cmap="binary")
plt.colorbar()
plt.savefig('mask.pdf')

plt.figure(3)
plt.imshow(np.abs(yfft),norm=colors.LogNorm(vmin=np.abs(yfft).min(), vmax=np.abs(yfft).max()), cmap="rainbow")
plt.colorbar()
plt.savefig('kspace.pdf')

plt.figure(4)
plt.imshow(image_abs, cmap="gray")
plt.colorbar()
plt.savefig('modelimage.pdf')


"""TV minimization with Split Bregman"""

#Calculate finite difference operator
D = [
    pylops.FirstDerivative(
        (ny, nx), axis=0, edge=False, kind="backward", dtype=np.complex128
    ),
    pylops.FirstDerivative(
        (ny, nx), axis=1, edge=False, kind="backward", dtype=np.complex128
    ),
]


# TV
#Reconstruct image from K-space
mu = 1/(mu_factor*(sigma * np.sqrt(12*np.log(nx*ny)))/np.sqrt(nxysub))
xinv = pylops.optimization.sparsity.splitbregman(
    1/np.sqrt(nxysub)*P * F,
    1/np.sqrt(nxysub)*y.ravel(),
    D,
    x0=image.ravel(),
    niter_outer=niter_out,
    niter_inner=niter_in,
    mu=mu,
    epsRL1s=lamda,
    tol=1e-30,
    tau=1.0,
    show=False
)[0]
xinv = xinv.reshape(ny,nx)
xinv_abs = np.abs(xinv)

"""Calculate SSIM and L2-difference"""
value = ssim(image_abs, xinv_abs, data_range=xinv_abs.max() - xinv_abs.min())
print('SSIM', value)

est_gt = xinv-image
real_est_gt = np.real(image - xinv)
#imag_est_gt = np.imag(image - xinv)
print("L2-Norm der real Differenz: ", np.linalg.norm(real_est_gt.ravel()))
#print("L2-Norm der imag Differenz: ", np.linalg.norm(imag_est_gt.ravel()))

l2_norm_diff = np.linalg.norm(est_gt)
print("L2-Norm der Differenz: ", l2_norm_diff)
print("L2-Norm ground truth", np.linalg.norm(image.ravel()))
print("Loo-Norm Differenz", np.linalg.norm(est_gt.flatten(), ord=np.inf))

"""Calculate unbiased estimator"""
residual = y-P*(F * xinv.ravel())

estimator_u = xinv + (1/nxysub*np.matmul(M,((P*F).H*residual).ravel())).reshape(ny,nx)

diff_gt_est_u = estimator_u - image
print("L2-Norm u-Differenz: ", np.linalg.norm(diff_gt_est_u.flatten()))
print("Loo-Norm u-Differenz: ", np.linalg.norm(diff_gt_est_u.flatten(), ord=np.inf))

"""Choose alpha, confidence level and calcualte delta and difference"""
alpha = 0.05

difference = np.sqrt(2*nxysub)*diff_gt_est_u.ravel()/(np.sqrt(diagonal)*sigma)

# calculate confidence radius delta
delta = sigma*np.sqrt(diagonal)*np.sqrt(np.log(1/alpha))/np.sqrt(nxysub)

"""Plot reconstruction, difference and unbiased estimate"""
plt.figure(6)
plt.imshow(xinv_abs, cmap="gray")
plt.colorbar()
plt.savefig('reconstruction.pdf')


plt.figure(7)
plt.imshow(np.abs(image - xinv), cmap="gray")
plt.colorbar()
plt.savefig('reconstructionminusgt.pdf')


fig, ax = plt.subplots()
extent = (-3, 4, -3, 4)
im = ax.imshow(np.abs(estimator_u), extent=extent, origin="upper", cmap='gray')
plt.colorbar(im, ax=ax, location='left')
x1, x2, y1, y2 = 2.4, 3.0, 2.4, 3.0  # subregion of the original image
axins = ax.inset_axes(
    [0.9, 0.8, 0.3, 0.3],
    xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
axins.imshow(np.abs(estimator_u), extent=extent, origin="upper", cmap='gray', norm=colors.LogNorm(vmin=(np.abs(estimator_u)).min(), vmax=(np.abs(estimator_u)).max()))

ax.indicate_inset_zoom(axins, edgecolor="white")
plt.savefig('unbiasedestimator.pdf')


plt.figure(8)
plt.imshow(np.abs(image - estimator_u), cmap="gray")
plt.colorbar()
plt.savefig('unbiasedestimatorminusgt.pdf')


"""QQ-Plot"""
difference_real = np.real(difference)
difference_imag = np.imag(difference)
difference_1D_real = difference_real.flatten()
difference_1D_imag = difference_imag.flatten()
diff_sorted_real = difference_1D_real[np.argsort(difference_1D_real)]
diff_sorted_imag = difference_1D_imag[np.argsort(difference_1D_imag)]

# plot qqplot and histogram of difference
plt.figure(15)
sm.qqplot(diff_sorted_real, line='45')
plt.axvline(x = -3, color = 'b')
plt.axvline(x = 3, color = 'b')
plt.savefig('qqplotrealpart.pdf')


"""Plot Confidence intervals for lines in image"""
image_abs_1D = image_abs.flatten()
estimator_u_abs = np.abs(estimator_u)


plt.figure(10)
plt.errorbar(range(nx), estimator_u_abs[78, :], yerr=np.abs(delta.reshape(nx,ny)[78,:]), capsize=2, marker='o', drawstyle='steps', linestyle='', markerfacecolor='none')
plt.plot(image_abs[78, :], 'r+')
plt.plot(xinv_abs[78, :], '+', color="orange")
plt.savefig('confidenceintervalsline78.pdf')

plt.figure(11)
plt.errorbar(range(nx), estimator_u_abs[123, :], yerr=np.abs(delta.reshape(nx,ny)[123,:]), capsize=2, marker='o', drawstyle='steps', linestyle='', markerfacecolor='none')
plt.plot(image_abs[123, :], 'm+')
plt.plot(xinv_abs[123, :], 'y+')
plt.savefig('confidenceintervalsline123.pdf')

plt.figure(12)
start1 = [0, 155]
end1 = [78, 78]
start2 = [0, 155]
end2 = [123, 123]
plt.plot(start1, end1, color="red", linewidth=3)
plt.plot(start2, end2, color="magenta", linewidth=3)
plt.imshow(image_abs, cmap="gray")
plt.savefig('imagewithlines.pdf')


plt.figure(13)
start1 = [0, 155]
end1 = [78, 78]
start2 = [0, 155]
end2 = [123, 123]
plt.plot(start1, end1, color="orange", linewidth=3)
plt.plot(start2, end2, color="yellow", linewidth=3)
plt.imshow(xinv_abs, cmap="gray")
plt.savefig('TVinversion_withlines.pdf')


"""Calculate R and W"""
step = P*(F * est_gt).ravel()
restterm = 1/nxysub*np.matmul(M,((P*F).H*step).ravel()).reshape(nx,ny) - est_gt
restterm = restterm.flatten()
print("L2-Norm Remainder term:", np.linalg.norm(restterm))
print('Loo-Norm Remainder term', np.linalg.norm(restterm, ord=np.inf))
idx_rest_sorted = np.argsort(np.real(restterm))
rest_sorted = restterm[idx_rest_sorted]


Gaussianterm = 1/nxysub*np.matmul(M,((P*F).H*n).ravel()).reshape(nx,ny)
Gaussianterm = Gaussianterm.flatten()
Gaussianterm_normalized = np.sqrt(2*nxysub)*np.real(Gaussianterm)/(np.sqrt(diagonal)*sigma)  #np.sqrt(covariance_A)/sigma cov = 1 in uniform case
print("L2-Norm Gauss term: ", np.linalg.norm(Gaussianterm))
print('Loo-Norm Gauss term', np.linalg.norm(Gaussianterm, ord=np.inf))
idx_Gauss_sorted = np.argsort(Gaussianterm_normalized)
Gauss_sorted = Gaussianterm_normalized[idx_Gauss_sorted]


testterm = Gaussianterm - restterm
print("test decomposition", np.linalg.norm(testterm - diff_gt_est_u.flatten()))



"""Compute hit rates"""
hitrate = 0
for k in range(0, nx*ny-1):
    if np.abs(estimator_u.ravel()[k]-image.ravel()[k]) <= delta[k]:
        hitrate += 1

cov = hitrate/len(image_abs_1D)
print("Cov", cov)

# compute hitrate on support
suppx =np.nonzero(image.ravel())[0]
print(suppx)
hitratesupp = 0
for k in range(0, len(suppx)):
    l = suppx[k]
    if np.abs(estimator_u.ravel()[l]-image.ravel()[l]) <= delta[l]:
        hitratesupp += 1

covs = hitratesupp/len(suppx)
print("CovS", covs)
