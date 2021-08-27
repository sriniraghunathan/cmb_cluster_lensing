# SPT CMB-cluster lensing

## Estimator reference: SPTpol CMB-cluster lensing using polarisation-only data. arXiv: [1907.08906](https://arxiv.org/abs/1907.08605)

## Example: 
* Look into this [notebook](https://github.com/sriniraghunathan/cmb_cluster_lensing/blob/main/scripts/cluster_lensing_example.ipynb) for a simple example.

## Steps for pipeline execution:
* Step 1: Creating cluster lensed or random unlensed CMB simulations.
  * python step1_create_sims.py *(for cluster lensed sims)*
  * python step1_create_sims.py -clusters_or_randoms randoms *(for random unlensed sims used for background subtraction)*
* Step 2: Create models.
  * python step2_gen_models.py -dataset_fname [path_to_cluster_lensed_cmb_sim_file_from_step1]
* Step 3: Estimate covariance matrix using JK approach / obtain likelihoods. This module also helps you to do tSZ subtraction automatically if cluster correalted tSZ signal is added to the lensed CMB simulations.
  * python step3_get_likelihoods.py -dataset_fname [path_to_cluster_lensed_cmb_sim_file_from_step1]
