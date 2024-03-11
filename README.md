# Project_UQ_TV

## File usage
The file `nodewise_lasso_compute_M.py` is used for constructing the precision matrix M via the nodewise lasso using FISTA.   
Before running the Split-Bregman files for image reconstruction. You need to construct the M with the nodewise LASSO and run the file `compute_sample_covariance.py` to create the sample covariance matrix.  


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
