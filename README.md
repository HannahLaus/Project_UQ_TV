# Imaging with Confidence

This is the code to the paper "Imaging with Confidence: Uncertainty Quantification for High-dimensional Undersampled MR Images".

## File usage
1. Run one of the files `create_radial_mask.py` and `create_spiral_mask.py` for creating your desired k-space undersampling masks.
2. Run the file `nodewise_lasso_compute_M.py`, in order to obtain the matrix M via the nodewise LASSO using FISTA (Algorithm 1 in our paper).
3. Run the file `compute_sample_covariance.py` to obtain the sample covariance matrix.

The in-vivo data used in this paper cannot be shared due to a confidentiality agreement since the files contain metadata from the patients. Therefore, if one needs to reproduce the experiments we recommend to do so with the Shepp-Logan phantom (lines 17 in `split_bregman_complex_2D.py` and 23 in `split_bregman_complex_2D_multiple_experiments.py`).
   
4. Run the file `split_bregman_complex_2D.py` to reproduce our experiments from the numerical section.
5. Run the file `split_bregman_complex_2D_multiple_experiments.py` to reproduce the experiments from 4. multiple times each time with a different realization of the noise and to create the estimated probability matrix (Figure 2f and 3f in the paper).

The reconstruction in steps 4 and 5 uses the Split-Bregman algorithm and is inspired by the total variation minimization algorithm implemented in Pylops (https://pylops.readthedocs.io/en/v1.18.0/gallery/plot_tvreg.html#sphx-glr-gallery-plot-tvreg-py).



## Requirements
The following packages were used for running the code.

`matplotlib` *(v3.8.3)*  
`numpy` *(v1.24.4)*  
`cupy-cuda11x` *(v13.0.0)*    
`python` *(v3.9.13)*  
`scikit-image` *(v0.19.2)*  
`phantominator` *(v0.7.0)*   
`pylops` *(v2.2.0)*   
`scipy` *(v1.9.1)*   
`statsmodels` *(v0.13.2)*    
`torch` *(v1.9.0+cu111)*
