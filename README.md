# Project_UQ_TV

This is the code to the paper ...

## File usage
The file `split_bregman_complex_2D.py` reconstructs the image from an undersampled k-space using the Split-Bregman algorithm. Afterwards, it does the debiasing step as described in our paper (Algorithm 2) and creates all the plots.
The file `split_bregman_complex_2D_multiple_experiments.py`does the same but for multiple experiements and it averages the results over all the experiments in the end. 
The file `nodewise_lasso_compute_M.py` is used for constructing the precision matrix M via the nodewise lasso using FISTA (Algorithm 1 in our paper), which is needed for running the above files. With the file `create_radial_mask.py`one can create a radial mask of different sizes and different undersampling rates.  
Further you need to run the file `compute_sample_covariance.py` to create the sample covariance matrix, before running the split_bregam_complex_2D files.


## Requirements
We used the following packages for running the code other versions might be useable as well.

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
