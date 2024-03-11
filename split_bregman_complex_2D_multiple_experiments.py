import matplotlib.pyplot as plt
import os
import numpy as np
import pylops
import matplotlib.colors as colors
import scipy.io
from phantominator import shepp_logan
from skimage.metrics import structural_similarity as ssim



"""Choose noise level"""
sigma = 0.15
""" Choose sampling type"""
sampling = "radial"
"""Choose alpha to obtain (1-alpha) Confidence intervals"""
alpha = 0.05
"""Choose lambda and number of iterations for Split Bregman"""
lamda = [0.05, 0.05]
niter_out = 2000
niter_in = 50

"""Load image and rescale"""
image = shepp_logan((156,156,5), MR=True,zlims=(-.25,.25))[0][:,:,0]
image = image/np.linalg.norm(image)
ny, nx = image.shape
image_abs = np.abs(image)

"""Determine the elements which are sampled"""
if sampling == "radial":
    samplecovariance = np.load("samplecov_radial_full_50per.dat",allow_pickle=True)  # exchange here for the right masked
    M = np.load("M_radial_50_mont_lambda_0.0035_cc_fista_1000epochs.dat", allow_pickle=True)
    Msamplecov = np.matmul(M, samplecovariance)
    diagonal = np.diagonal(np.matmul(np.matmul(M, samplecovariance), np.conjugate(np.transpose(M))))
    mask = np.load("radial_full_mask_83_50per.dat", allow_pickle=True)
elif sampling == "spiral":
    samplecovariance = np.load("samplecov_spiral_full_43per.dat", allow_pickle=True)  #exchange here for the right masked
    M = np.load("M_spiral_40_mont_lambda_0.0035_cc_fista_1000epochs.dat", allow_pickle=True)
    Msamplecov = np.matmul(M, samplecovariance)
    diagonal = np.diagonal(np.matmul(np.matmul(M, samplecovariance), np.conjugate(np.transpose(M))))
    #mask = np.load("radial_full_mask_83_50per.dat", allow_pickle=True)
    mask = scipy.io.loadmat("k_space_mask_cart_w_edges_spiral_undersampling_factor_3.mat")
    mask = mask['k_space_mask']
    mask = mask[50:206,50:206]
else:
    raise Exception

iava = np.nonzero(mask.flatten())[0]

"""Define 2D-FFT and undersampling/restriction operator"""
Rop = pylops.Restriction(nx*ny, iava, axis=0, dtype=np.complex128)
nxysub = Rop.shape[0]
Fop = pylops.signalprocessing.FFT2D(dims=(ny, nx), norm="none", fftshift_after=True, ifftshift_before=True)

###############################################################################
# Apply Split Bregman with anisotropic TV regularization (aka sum of L1 norms of the
# first derivatives over x and y) for reconstruction of the image from the measurements,
# define the first derivative operator here.
#
# .. math::
#         J = \mu/2 ||\mathbf{y} - \mathbf{R} \mathbf{F} \mathbf{x}||_2
#         + || \nabla_x \mathbf{x}||_1 + || \nabla_y \mathbf{x}||_1

Dop = [
    pylops.FirstDerivative(
        (ny, nx), axis=0, edge=False, kind="backward", dtype=np.complex128
    ),
    pylops.FirstDerivative(
        (ny, nx), axis=1, edge=False, kind="backward", dtype=np.complex128
    ),
]


# Define all other parameters needed for Split Bregman here

mu = 1/(20*(sigma * np.sqrt(12*np.log(nx*ny)))/np.sqrt(nxysub))
print('mu', mu)

hitrate_all =[]
hitrate_all_support = []
error_unbiased_L2 = []
error_debiased_L2 = []
error_unbiased_Linf = []
error_debiased_Linf = []
R_inf_M = []
R_L2_M = []
W_inf_M = []
W_L2_M = []
SSIM = []
rel_noise = []
num_exp = 100
est_prob_matrix = np.zeros((nx*ny))

"""Start loop over number of experiments"""
for j in range(num_exp):
    print(j)
    """Sample the noise given the variance sigma and calulate the measurements y"""
    n_full = sigma/(np.sqrt(2)) * np.random.normal(0, 1, (ny, nx, 2)).view(np.complex128)[:,:,0]
    n = Rop * n_full.ravel()
    y = Rop * ((Fop * image).ravel() + n_full.ravel())
    rel_noise.append(np.linalg.norm(n)/np.linalg.norm(Rop * Fop * image.ravel()))
    print('Noise level:', np.linalg.norm(n)/np.linalg.norm(Rop * Fop * image.ravel()))

    """Apply Split Bregman and reconstruct the image."""
    xinv = pylops.optimization.sparsity.splitbregman(
        1/np.sqrt(nxysub)*Rop * Fop,
        1/np.sqrt(nxysub)*y.ravel(),
        Dop,
        niter_outer=niter_out,
        niter_inner=niter_in,
        mu=mu,
        epsRL1s=lamda,
        tol=1e-30,
        tau=1.0,
        show=False,
    )[0]
    xinv = xinv.reshape(ny,nx)
    xinv_abs = np.abs(xinv)
    value = ssim(image_abs, xinv_abs, data_range=xinv_abs.max() - xinv_abs.min())
    SSIM.append(value)
    print(value)


    """Calculate the difference between the reconstruction and the image"""
    est_gt = xinv-image
    real_est_gt = np.real(image - xinv)
    #imag_est_gt = np.imag(image - xinv)
    print("L2-Norm der real Differenz: ", np.linalg.norm(real_est_gt.ravel()))
    #print("L2-Norm der imag Differenz: ", np.linalg.norm(imag_est_gt.ravel()))

    l2_norm_diff = np.linalg.norm(est_gt)
    print("L2-Norm der Differenz: ", l2_norm_diff)
    error_unbiased_L2.append(l2_norm_diff)
    print("L2-Norm ground truth", np.linalg.norm(image.ravel()))
    print("Loo-Norm Differenz", np.linalg.norm(est_gt.flatten(), ord=np.inf))
    error_unbiased_Linf.append(np.linalg.norm(est_gt.flatten(), ord=np.inf))


    """Calculate the residual, M and the unbiased estimator"""
    residual = y-Rop*(Fop * xinv.ravel())

    estimator_u = xinv + (1/nxysub*np.matmul(M,((Rop*Fop).H*residual).ravel())).reshape(ny,nx)

    diff_gt_est_u = estimator_u - image
    print("L2-Norm u-Differenz: ", np.linalg.norm(diff_gt_est_u.flatten()))
    print("Loo-Norm u-Differenz: ", np.linalg.norm(diff_gt_est_u.flatten(), ord=np.inf))
    error_debiased_L2.append(np.linalg.norm(diff_gt_est_u.flatten()))
    error_debiased_Linf.append(np.linalg.norm(diff_gt_est_u.flatten(), ord=np.inf))

    """calculate confidence radius delta """
    delta = sigma*np.sqrt(diagonal)*np.sqrt(np.log(1/alpha))/np.sqrt(nxysub)

    print('delta', delta)


    """Calculater Remainder term with M"""
    step = Rop*(Fop.dot(est_gt.ravel()))
    restterm = 1/nxysub*np.matmul(M,((Rop*Fop).H*step).ravel()).reshape(nx,ny) - est_gt
    print("L2-Norm Remainder term:", np.linalg.norm(restterm))
    print('Loo-Norm Remainder term', np.linalg.norm(restterm.ravel(), ord=np.inf))
    R_L2_M.append(np.linalg.norm(restterm))
    R_inf_M.append( np.linalg.norm(restterm.ravel(), ord=np.inf))
    restterm = restterm.flatten()

    """Calculate Gauss term with M"""
    Gaussianterm = 1/nxysub*np.matmul(M,((Rop*Fop).H*n).ravel()).reshape(nx,ny)
    Gaussianterm = Gaussianterm.flatten()
    Gaussianterm_normalized = np.sqrt(2*nxysub)*np.real(Gaussianterm)/(np.sqrt(diagonal)*sigma)  #np.sqrt(covariance_A)/sigma cov = 1 in uniform case
    print("L2-Norm Gauss term: ", np.linalg.norm(Gaussianterm))
    print('Loo-Norm Gauss term', np.linalg.norm(Gaussianterm, ord=np.inf))
    W_L2_M.append(np.linalg.norm(Gaussianterm))
    W_inf_M.append(np.linalg.norm(Gaussianterm, ord=np.inf))

    """Calculate difference between Gaussianterm and restterm"""
    testterm = Gaussianterm - restterm
    print("test decomposition", np.linalg.norm(testterm - diff_gt_est_u.flatten()))


    """compute hitrate """
    image_abs_1D = image_abs.flatten()
    hitrate = 0
    for k in range(0, nx*ny-1):
        if np.abs(estimator_u.ravel()[k]-image.ravel()[k]) <= delta[k]:
            hitrate += 1
            est_prob_matrix[k] += 1

    cov = hitrate/len(image_abs_1D)
    print("Cov", cov)
    hitrate_all.append(cov)

    """ compute hitrate on support """
    suppx =np.nonzero(image.ravel())[0]
    #print(suppx)
    hitratesupp = 0
    for k in range(0, len(suppx)):
        l = suppx[k]
        if np.abs(estimator_u.ravel()[l]-image.ravel()[l]) <= delta[l]:
            hitratesupp += 1

    covs = hitratesupp/len(suppx)
    hitrate_all_support.append(covs)
    print("CovS", covs)


""""Print the Average errors, normed R and Ws"""
print('########## Average results #######')
print('Number of Experiments:', num_exp )
print('L2-Norm Differenz mean:', sum(error_unbiased_L2)/len(error_unbiased_L2))
print('Loo-Norm Differenz mean:', sum(error_unbiased_Linf)/len(error_unbiased_Linf))
print('L2-Norm u-Differenz mean:',  sum(error_debiased_L2)/(len(error_debiased_L2)))
print('Loo-Norm u-Differenz mean:', sum(error_debiased_Linf)/len(error_debiased_Linf))
print()
print('Norms for W and R with M')
print('L2-Norm Remainder term with M mean:', sum(R_L2_M)/len(R_L2_M))
print('Loo-Norm Remainder term with M mean:', sum(R_inf_M)/len(R_inf_M))
print('L2-Norm Gauss term with M mean:', sum(W_L2_M)/(len(W_L2_M)))
print('Loo-Norm Gauss term with M mean:', sum(W_inf_M)/len(W_inf_M))
print()
print('Hitrates, noise level and SSIM')
print('hitrate mean:', sum(hitrate_all)/(len(hitrate_all)))
print('hitrate support mean:', sum(hitrate_all_support)/(len(hitrate_all_support)))
print('relative noise:', sum(rel_noise)/len(rel_noise))
print('SSIM:', sum(SSIM)/len(rel_noise))
est_prob_matrix.dump("est_prob_matrix_spiral_60per_sigma0.15.dat", protocol=4)
print(est_prob_matrix)


"Plot the estimated probability matrix"
est_prob_matrix = est_prob_matrix.reshape(nx,nx)
plt.imshow(image/100,  norm=colors.Normalize(0.8,1))
plt.colorbar()
plt.savefig('empprobmatrix_spiral_60_12pernoise.pdf')
plt.show()
