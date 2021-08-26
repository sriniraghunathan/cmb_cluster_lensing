#!/usr/bin/env python
########################

########################
#load desired modules
import numpy as np, sys, os, scipy as sc, argparse, glob
sys_path_folder='/Users/sraghunathan/Research/SPTPol/analysis/git/cmb_cluster_lensing/python/'
sys.path.append(sys_path_folder)

import flatsky, tools, lensing, foregrounds, misc

from tqdm import tqdm

from pylab import *
cmap = cm.RdYlBu_r

import warnings
warnings.filterwarnings('ignore',category=RuntimeWarning)
print('\n')
########################

########################
parser = argparse.ArgumentParser(description='')
parser.add_argument('-dataset_fname', dest='dataset_fname', action='store', help='dataset_fname', type=str, default='../results//nx120_dx1/beam1.2/noise5/10amcutouts/T/clusters_700objects_25sims0to25.npy')


args = parser.parse_args()
args_keys = args.__dict__
for kargs in args_keys:
    param_value = args_keys[kargs]

    if isinstance(param_value, str):
        cmd = '%s = "%s"' %(kargs, param_value)
    else:
        cmd = '%s = %s' %(kargs, param_value)
    exec(cmd)

########################

########################
data = np.load(dataset_fname, allow_pickle= True).item()
param_dict = data['param_dict']
#cutouts specs 
dx = param_dict['dx'] #pixel resolution in arcmins
pol = param_dict['pol']
if not pol:
    tqulen = 1
else:
    tqulen = 3
tqu_tit_arr = ['T', 'Q', 'U']
cutout_size_am = param_dict['cutout_size_am'] #arcmins
x1, x2 = -cutout_size_am/2. * dx, cutout_size_am/2. *dx

#read mock data
data_stack_dic = data['clusters']['stack']

#get and read random stack
fd = '/'.join( dataset_fname.split('/')[:-1] )
random_dataset_fname = glob.glob( '%s/randoms*' %(fd) )[0]
random_data = np.load(random_dataset_fname, allow_pickle= True).item()['randoms']
random_stack_dic = random_data['stack']
random_stack = random_stack_dic[0]

#subtract background from data stack
for keyname in data_stack_dic:
    data_stack_dic[keyname] -= random_stack
    if (0):
        for tqu in range(len(data_stack_dic[keyname])):
            subplot(1, tqulen, tqu+1); imshow(data_stack_dic[keyname][tqu], cmap=cmap, extent = [x1, x2, x1, x2]); colorbar()
            title('%s' %(tqu_tit_arr[tqu]))
        show(); sys.exit()
    #print(data_stack_dic[keyname].shape)

#get models
model_fd = '%s/models/' %(fd)
model_flist = sorted( glob.glob('%s/*.npy' %(model_fd)) )
model_dic = {}
for model_fname in model_flist:
    model_data = np.load(model_fname, allow_pickle=True).item()
    model_arr = np.asarray( list(model_data.values()) )
    model = np.mean(model_arr, axis = 0)
    model_keyname_tmp = '_'.join(model_fname.split('_')[-2:]).replace('mass', '').replace('z','').replace('.npy','').split('_')
    model_mass, model_z = float(model_keyname_tmp[0]), float(model_keyname_tmp[1])
    model_keyname = ( round(model_mass, 3), round(model_z, 3) )
    model_dic[model_keyname] = model

#subtract M=0 from all
bg_model_keyname = (0., 0.7)
for model_keyname in model_dic:
    if model_keyname == bg_model_keyname: continue
    model_dic[model_keyname] -= model_dic[bg_model_keyname]
    if (0):
        for tqu in range(len(model_dic[model_keyname])):
            subplot(1, tqulen, tqu+1); imshow(model_dic[model_keyname][tqu], cmap=cmap, extent = [x1, x2, x1, x2], vmin = -2., vmax = 2.); 
            colorbar()
            title('(%s, %s): %s' %(model_keyname[0], model_keyname[1], tqu_tit_arr[tqu]))
            axhline(lw = 0.5); axvline(lw = 0.5)
        show(); 

if (1):
    #params or supply a params file
    noiseval = param_dict['noiseval'] #uK-arcmin
    beamval = param_dict['beamval'] #arcmins
    #foregrounds
    try:
        fg_gaussian = param_dict['fg_gaussian'] #Gaussian realisation of all foregrounds
    except:
        fg_gaussian = False



    #sim stuffs
    total_clusters = param_dict['total_clusters']

    #cluster info
    cluster_mass = param_dict['cluster_mass']
    cluster_z = param_dict['cluster_z']

    #cluster mass definitions
    delta=param_dict['delta']
    rho_def=param_dict['rho_def']
    profile_name=param_dict['profile_name']

########################
########################
#get JK based covariance from cluster cuouts
dummysimcntr = 1
cluster_cutouts_rotated_arr=data['clusters']['cutouts_rotated'][dummysimcntr] - random_stack
if (0):
    tmp_stack=np.mean(cluster_cutouts_rotated_arr, axis=0)
    for tqu in range(len(tmp_stack)):
        subplot(1, tqulen, tqu+1); imshow(tmp_stack[tqu], cmap=cmap, extent = [x1, x2, x1, x2]); colorbar()
    show(); sys.exit()
cluster_grad_mag_arr=data['clusters']['grad_mag'][dummysimcntr]

if param_dict['howmany_jk_samples']<500:
    param_dict['howmany_jk_samples'] = 500

jk_cov=tools.get_jk_covariance(cluster_cutouts_rotated_arr, param_dict['howmany_jk_samples'], weights=cluster_grad_mag_arr, only_T=True)
if (0):
    print(jk_cov.shape)
    clf(); imshow(jk_cov, cmap=cmap); colorbar(); show(); #sys.exit()

########################
########################
#get likelihoods
master_loglarr = []
for simcntr in sorted(data_stack_dic):
    for tqu in range(tqulen):
        loglarr = []
        massarr = []
        data_vec = data_stack_dic[simcntr][tqu].flatten()
        for model_keyname in sorted( model_dic ):
            model_vec = model_dic[model_keyname][tqu].flatten()
            loglval = tools.get_lnlikelihood(data_vec, model_vec, jk_cov)
            loglarr.append( loglval )
            massarr.append( model_keyname[0] )
        massarr = np.asarray( massarr )
        massarr, larr, recov_mass, snr = tools.lnlike_to_like(massarr, loglarr)
        #logl_dic[simcntr] = [massarr, loglarr, larr]
        master_loglarr.append( loglarr )
        plot(massarr, larr, label = simcntr, lw = 0.5);

combined_loglarr = np.sum(master_loglarr, axis = 0)
massarr, combined_larr, combined_recov_mass, combined_snr = tools.lnlike_to_like(massarr, combined_loglarr)
plot(massarr, combined_larr, lw = 1.5, color = 'black', label = r'Combined');
legend(loc = 4, ncol = 4, fontsize = 8)
xlabel(r'M$_{200m}$ [$10^{14}$M$_{\odot}$]', fontsize = 14)
ylabel(r'Normalised $\mathcal{L}$', fontsize = 14)
axvline(cluster_mass/1e14, ls = '-.', lw = 2.)
title(r'%s clusters; $\Delta_{\rm T} = %s \mu{\rm K-arcmin}$' %(total_clusters, noiseval))
show(); 
sys.exit()

