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
parser.add_argument('-dataset_fname', dest='dataset_fname', action='store', help='dataset_fname', type=str, default='../results//nx120_dx1/beam1.2/noise5/10amcutouts/nogaussianfg/T/clusters_700objects_10sims0to10.npy')
parser.add_argument('-use_1d', dest='use_1d', action='store', help='use_1d', type=int, default=0)
parser.add_argument('-totiters_for_model', dest='totiters_for_model', action='store', help='totiters_for_model', type=int, default=1)#25)


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
ny = nx = int(cutout_size_am / dx)
x1, x2 = -cutout_size_am/2., cutout_size_am/2.
#sim stuffs
total_clusters = param_dict['total_clusters']

##########################################
##########################################
#read mock data + remove tsz estimate if necessary
try:
    add_cluster_tsz = param_dict['add_cluster_tsz']
except:
    add_cluster_tsz = False
try:
    add_cluster_ksz = param_dict['add_cluster_ksz']
except:
    add_cluster_ksz = False

if not add_cluster_tsz:
    data_stack_dic = data['clusters']['stack']
else: #handle tsz
    #stack rotated cutouts + apply gradient magnitude weights
    data_stack_dic = {}
    totsims = len(data['clusters']['cutouts_rotated'])
    for simcntr in range( totsims ):
        cutouts_rotated_arr=data['clusters']['cutouts_rotated'][simcntr]
        grad_mag_arr=data['clusters']['grad_mag'][simcntr]

        ###stack = tools.stack_rotated_tqu_cutouts(cutouts_rotated_arr, weights_for_cutouts = grad_mag_arr)

        #estimate and remove tSZ from rotated stack
        cutouts_rotated_arr_for_tsz_estimation = np.copy(cutouts_rotated_arr)
        tsz_estimate = tools.stack_rotated_tqu_cutouts(cutouts_rotated_arr_for_tsz_estimation, weights_for_cutouts = None, perform_random_rotation = True)
        
        #fit tsz model
        tsz_fit_model = foregrounds.fit_fot_tsz(tsz_estimate[0], dx)

        if (0):
            subplot(131); imshow(tsz_estimate[0], cmap=cmap, extent = [x1, x2, x1, x2]); colorbar(); 
            subplot(132); imshow(tsz_fit_model, cmap=cmap, extent = [x1, x2, x1, x2]); colorbar(); 
            subplot(133); imshow(tsz_estimate[0]-tsz_fit_model, cmap=cmap, extent = [x1, x2, x1, x2]); colorbar(); 
            show(); sys.exit()

        if (1):
            print('\n\t\t\tfitting for tsz\n\n')
            tsz_estimate[0] = np.copy(tsz_fit_model)

        cutouts_rotated_arr[:,0] -= tsz_estimate[0]
        data['clusters']['cutouts_rotated'][simcntr] = cutouts_rotated_arr
        stack_after_tsz_removal = tools.stack_rotated_tqu_cutouts(cutouts_rotated_arr, weights_for_cutouts = grad_mag_arr)

        if (0):
            subplot(131); imshow(stack[0], cmap=cmap, extent = [x1, x2, x1, x2]); colorbar(); 
            subplot(132); imshow(tsz_estimate[0], cmap=cmap, extent = [x1, x2, x1, x2]); colorbar();
            subplot(133); imshow(stack_after_tsz_removal[0], cmap=cmap, extent = [x1, x2, x1, x2]); colorbar(); show(); sys.exit()

        stack = np.copy(stack_after_tsz_removal)

        data_stack_dic[simcntr]=stack
##########################################
##########################################

#get and read random stack
fd = '/'.join( dataset_fname.split('/')[:-1] )
random_dataset_fname = glob.glob( '%s/randoms*' %(fd) )[0]
random_data = np.load(random_dataset_fname, allow_pickle= True).item()['randoms']
random_stack_dic = random_data['stack']
random_stack = random_stack_dic[0]

#subtract background from data stack
for keyname in data_stack_dic:
    if (1):
        tmp_stack = data_stack_dic[keyname]
        tmp_stack_bg_sub = data_stack_dic[keyname] - random_stack
        sbpl=1
        for tqu in range(len(data_stack_dic[keyname])):
            subplot(tqulen, 3, sbpl); imshow(tmp_stack[tqu], cmap=cmap, extent = [x1, x2, x1, x2]); colorbar(); sbpl+=1
            subplot(tqulen, 3, sbpl); imshow(random_stack[tqu], cmap=cmap, extent = [x1, x2, x1, x2]); colorbar(); sbpl+=1
            subplot(tqulen, 3, sbpl); imshow(tmp_stack_bg_sub[tqu], cmap=cmap, extent = [x1, x2, x1, x2], vmin = -2.5, vmax = 2.5); colorbar(); sbpl+=1
            title('%s' %(tqu_tit_arr[tqu]))
        show(); sys.exit()
    data_stack_dic[keyname] -= random_stack
    data_stack_dic[keyname][np.isnan(data_stack_dic[keyname])] = 0.
    #print(data_stack_dic[keyname].shape)
    if np.sum(data_stack_dic[keyname][1]) == 0.: tqulen = 1
##########################################
##########################################

#get models
model_fd = '%s/models/' %(fd)
if not os.path.exists(model_fd):
    tmp_fd = fd.replace('_withclusterksz','').replace('_withclustertsz','')
    model_fd = '%s/models/' %(tmp_fd)
model_flist = sorted( glob.glob('%s/*_%ssims_*.npy' %(model_fd, totiters_for_model)) )

def get_model_keyname(model_fname):
    model_keyname_tmp = '_'.join(model_fname.split('_')[-3:-1]).replace('mass', '').replace('z','').replace('.npy','').split('_')
    model_mass, model_z = float(model_keyname_tmp[0]), float(model_keyname_tmp[1])
    model_keyname = ( round(model_mass, 3), round(model_z, 3) )
    return model_keyname   

#get gradient orientation first for each cluster in each (M,z) for each sim.
#next for every cluster we will get the median grad orientation across all (M,z) for each sim.
#this ensures that we rotate a given cluster lensed by all (M,z) by the same angle. Otherwise, there likelihoods can be shaky.
tmp_model_orien_dic = {}
for model_fname in model_flist:
    model_data = np.load(model_fname, allow_pickle=True).item()
    for simkeyname in model_data:
        if simkeyname not in tmp_model_orien_dic:
            tmp_model_orien_dic[simkeyname] = []
        cutouts_rotated_arr, grad_mag_arr, grad_orien_arr = model_data[simkeyname]['cutouts']
        tmp_model_orien_dic[simkeyname].append(grad_orien_arr)

#final gradient orientation is obtained in this step
model_orien_dic = {}
for simkeyname in tmp_model_orien_dic:
    model_orien_dic[simkeyname] = []
    grad_orien_arr = np.asarray( tmp_model_orien_dic[simkeyname] ) #vector with dimensions total_models x total_clusters x tqulen
    model_orien_dic[simkeyname] = np.mean(grad_orien_arr, axis = 0) #vector with dimensions total_clusters x tqulen

#models are computed here by rotating each cluster along the orientations estimated above
model_dic = {}
for model_fname in model_flist:
    model_data = np.load(model_fname, allow_pickle=True).item()
    '''
    #model_arr = np.asarray( list(model_data.values()) )
    model_arr = []
    for simkeyname in model_data:
        model_arr.append(model_data[simkeyname]['stack'])
    '''
    model_arr = []
    for simkeyname in model_data:
        cutouts_arr, grad_mag_arr, grad_orien_arr = model_data[simkeyname]['cutouts']
        grad_orien_arr_avg = model_orien_dic[simkeyname]

        cutouts_rotated_arr = tools.get_rotated_tqu_cutouts_simple(cutouts_arr, grad_orien_arr_avg, total_clusters, tqulen)
        stack=tools.stack_rotated_tqu_cutouts(cutouts_rotated_arr, weights_for_cutouts = grad_mag_arr)
        model_arr.append( stack )

    model = np.mean(model_arr, axis = 0)
    model_keyname = get_model_keyname(model_fname)
    model_dic[model_keyname] = model

##########################################
##########################################

#subtract M=0 from all
bg_model_keyname = (0., 0.5)#0.7)
for model_keyname in model_dic:
    if model_keyname == bg_model_keyname: continue
    model_dic[model_keyname] -= model_dic[bg_model_keyname]
    if (0):#model_keyname[0]>0.8 and model_keyname[0]<1.2:#(1):
        #for tqu in range(len(model_dic[model_keyname])):
        for tqu in range(tqulen):
            print(model_dic[model_keyname][tqu])
            vmin, vmax = -2., 2.
            vmin, vmax = None, None
            subplot(1, tqulen, tqu+1); imshow(model_dic[model_keyname][tqu], cmap=cmap, extent = [x1, x2, x1, x2], vmin = vmin, vmax = vmax); 
            colorbar()
            title('(%s, %s): %s' %(model_keyname[0], model_keyname[1], tqu_tit_arr[tqu]))
            axhline(lw = 0.5); axvline(lw = 0.5)
        show(); sys.exit()
model_dic[bg_model_keyname] -= model_dic[bg_model_keyname]
#sys.exit()

##########################################
##########################################

if (1):
    #params or supply a params file
    noiseval = param_dict['noiseval'] #uK-arcmin
    beamval = param_dict['beamval'] #arcmins
    #foregrounds
    try:
        fg_gaussian = param_dict['fg_gaussian'] #Gaussian realisation of all foregrounds
    except:
        fg_gaussian = False

    #ILC
    try:
        ilc_file = param_dict['ilc_file'] #ILC residuals
        which_ilc = param_dict['which_ilc']
    except:
        ilc_file = None
        which_ilc = None

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
if use_1d:
    cluster_cutouts_rotated_arr_1d = np.zeros((total_clusters, tqulen, nx))
    for i in range(total_clusters):
        for tqu in range(tqulen):
            cluster_cutouts_rotated_arr_1d[i, tqu] = np.mean(cluster_cutouts_rotated_arr[i, tqu], axis = 0)
    cluster_cutouts_rotated_arr = np.copy( cluster_cutouts_rotated_arr_1d )
cluster_grad_mag_arr=data['clusters']['grad_mag'][dummysimcntr]

min_howmany_jk_samples = int(total_clusters * 0.9)
try:
    howmany_jk_samples = param_dict['howmany_jk_samples']
except:
    howmany_jk_samples = min_howmany_jk_samples
if howmany_jk_samples<min_howmany_jk_samples:
    howmany_jk_samples = min_howmany_jk_samples
if howmany_jk_samples>len(cluster_cutouts_rotated_arr):
    howmany_jk_samples = len(cluster_cutouts_rotated_arr) - 1
#np.random.seed(100)
jk_cov_T=tools.get_jk_covariance(cluster_cutouts_rotated_arr, howmany_jk_samples, weights=cluster_grad_mag_arr, T_or_Q_or_U='T')
jk_cov_dic = {}
jk_cov_dic['T'] = jk_cov_T
if pol:
    jk_cov_Q=tools.get_jk_covariance(cluster_cutouts_rotated_arr, howmany_jk_samples, weights=cluster_grad_mag_arr, T_or_Q_or_U='Q')
    jk_cov_U=tools.get_jk_covariance(cluster_cutouts_rotated_arr, howmany_jk_samples, weights=cluster_grad_mag_arr, T_or_Q_or_U='U')
    jk_cov_dic['Q'] = jk_cov_Q
    jk_cov_dic['U'] = jk_cov_U
    #jk_cov_all=tools.get_jk_covariance(cluster_cutouts_rotated_arr, howmany_jk_samples, weights=cluster_grad_mag_arr, T_or_Q_or_U='all')
    #jk_cov_dic['all'] = jk_cov_all

if (0):
    clf(); 
    subplot(221);imshow(jk_cov_T, cmap=cmap); colorbar(); 
    subplot(222);imshow(jk_cov_Q, cmap=cmap); colorbar(); 
    subplot(223);imshow(jk_cov_U, cmap=cmap); colorbar(); 
    subplot(224);imshow(jk_cov_all, cmap=cmap); colorbar(); show(); sys.exit()

########################
########################
#get likelihoods
def get_plname():
    plfolder = '%s/plots/' %(dataset_fd)
    if not os.path.exists(plfolder): os.system('mkdir -p %s' %(plfolder))
    plname = '%s/%sclusters_beam%s_noise%s' %(plfolder, total_clusters, beamval, noiseval)
    if tqulen == 3:
        plname = '%s_TQU' %(plname)
    else:
        plname = '%s_T' %(plname)

    if fg_gaussian:
        titstr = 'FG added'
        plname = '%s_withfg' %(plname)
    else:
        titstr = 'No FG'
        plname = '%s_nofg' %(plname)

    if ilc_file is not None:
        titstr = 'ILC: %s' %(which_ilc)
        plname = '%s_ilc_%s' %(plname, which_ilc)

    if add_cluster_tsz:
        plname = '%s_withclustertsz' %(plname)
        titstr = '%s + cluster tSZ' %(titstr)
    if add_cluster_ksz:
        plname = '%s_withclusterksz' %(plname)
        titstr = '%s + cluster kSZ' %(titstr)

    rsval_used = dataset_fname.split('_')[-1].replace('.npy', '')

    plname = '%s_%s' %(plname.replace('plots//', 'plots/'), rsval_used)
    opfname = '%s.npy' %(plname.replace('/plots/', '/results_'))
    plname = '%s.png' %(plname)
    
    return plname, opfname, titstr

res_dic = {}
res_dic['likelihood'] = {}
testing = 0
tr, tc = tqulen, 1
tqudic = {0: 'T', 1: 'Q', 2: 'U'}
for tqu in range(tqulen):
    res_dic['likelihood'][tqudic[tqu]] = {}
    master_loglarr = []
    ax = subplot(tr, tc, tqu+1)
    for simcntr in sorted(data_stack_dic):
        loglarr = []
        massarr = []
        if use_1d:
            data_vec = np.mean(data_stack_dic[simcntr][tqu], axis = 0)
        else:
            data_vec = data_stack_dic[simcntr][tqu].flatten()
        if testing:colorarr = [cm.jet(int(d)) for d in np.linspace(0, 255, len(model_dic))]
        for modelcntr, model_keyname in enumerate( sorted( model_dic ) ):
            if model_keyname[0]>4.0: continue
            if use_1d:
                model_vec = np.mean(model_dic[model_keyname][tqu], axis = 0)
                if testing: plot(model_vec, color = colorarr[modelcntr])
            else:
                model_vec = model_dic[model_keyname][tqu].flatten()
            loglval = tools.get_lnlikelihood(data_vec, model_vec, jk_cov_dic[tqudic[tqu]])
            loglarr.append( loglval )
            massarr.append( model_keyname[0] )
        if testing: show(); sys.exit()
        massarr = np.asarray( massarr )
        massarr_mod, larr, recov_mass, snr = tools.lnlike_to_like(massarr, loglarr)
        #logl_dic[simcntr] = [massarr, loglarr, larr]
        master_loglarr.append( loglarr )
        plot(massarr_mod, larr, label = simcntr, lw = 0.5);
        res_dic['likelihood'][tqudic[tqu]][simcntr] = [massarr, loglarr, massarr_mod, larr, recov_mass, snr]

    combined_loglarr = np.sum(master_loglarr, axis = 0)
    massarr_mod, combined_larr, combined_recov_mass, combined_snr = tools.lnlike_to_like(massarr, combined_loglarr)
    combined_mean_mass, combined_mean_mass_low_err, combined_mean_mass_high_err = tools.get_width_from_sampling(massarr_mod, combined_larr)
    combined_mean_mass_err = (combined_mean_mass_low_err + combined_mean_mass_high_err)/2.
    plot(massarr_mod, combined_larr, lw = 1.5, color = 'black', label = r'Combined: %.2f $\pm$ %.2f' %(combined_mean_mass, combined_mean_mass_err));
    axvline(cluster_mass/1e14, ls = '-.', lw = 2.)
    dataset_fd = '/'.join(dataset_fname.split('/')[:-1])
    plname, opfname, titstr = get_plname()
    res_dic['likelihood'][tqudic[tqu]][-1] = [massarr, combined_loglarr, massarr_mod, combined_larr, combined_recov_mass, combined_snr]
    
    if tqu == 0:
        if tqulen == 1:
            legend(loc = 4, ncol = 4, fontsize = 8)
        else:
            legend(loc = 4, ncol = 8, fontsize = 6)
    if tqu+1 == tqulen:
        mdefstr = 'M$_{%s%s}$' %(param_dict['delta'], param_dict['rho_def'])
        xlabel(r'%s [$10^{14}$M$_{\odot}$]' %(mdefstr), fontsize = 14)
    ylabel(r'Normalised $\mathcal{L}$', fontsize = 14)
    title(r'%s clusters (SNR = %.2f); $\Delta_{\rm T} = %s \mu{\rm K-arcmin}$; %s' %(total_clusters, combined_snr, noiseval, titstr), fontsize = 10)
res_dic['param_dict'] = param_dict
np.save(opfname, res_dic)
##savefig(plname)
show();
print(plname)
sys.exit()

