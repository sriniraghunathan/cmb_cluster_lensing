#!/usr/bin/env python
########################

########################
#load desired modules
import numpy as np, sys, os, scipy as sc, argparse
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
parser.add_argument('-minM', dest='minM', action='store', help='minM', type=float, default=0.)
parser.add_argument('-maxM', dest='maxM', action='store', help='maxM', type=float, default=5.)
parser.add_argument('-delM', dest='delM', action='store', help='delM', type=float, default=0.1)
parser.add_argument('-totiters_for_model', dest='totiters_for_model', action='store', help='totiters_for_model', type=int, default=1)
parser.add_argument('-random_seed_for_models', dest='random_seed_for_models', action='store', help='random_seed_for_models', type=int, default=100)

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

data_folder = param_dict['data_folder']
results_folder = param_dict['results_folder']

#params or supply a params file
dx = param_dict['dx'] #pixel resolution in arcmins
boxsize_am = param_dict['boxsize_am'] #boxsize in arcmins
nx = int(boxsize_am/dx)
mapparams = [nx, nx, dx]
x1,x2 = -nx/2. * dx, nx/2. * dx
verbose = 0
pol = param_dict['pol']
debug = param_dict['debug']

#beam and noise levels
noiseval = param_dict['noiseval'] #uK-arcmin
if pol:
    noiseval = [noiseval, noiseval * np.sqrt(2.), noiseval * np.sqrt(2.)]
beamval = param_dict['beamval'] #arcmins

#foregrounds
try:
    fg_gaussian = param_dict['fg_gaussian'] #Gaussian realisation of all foregrounds
except:
    fg_gaussian = False

try:
    add_cluster_tsz=param_dict['add_cluster_tsz']
except:
    add_cluster_tsz=False

try:
    add_cluster_ksz=param_dict['add_cluster_ksz']
except:
    add_cluster_ksz=False

#ILC
try:
    ilc_file = param_dict['ilc_file'] #ILC residuals
    which_ilc = param_dict['which_ilc']
except:
    ilc_file = None
    which_ilc = None

if ilc_file is not None:
    fg_gaussian = None
    if which_ilc == 'cmbtszfree':
        add_cluster_tsz = None
    else:
        print('\n\n\tyou have requested a ILC that is not tSZ-free. Weighted tsz is not implemented yet. aborting script here.')
        sys.exit()

#CMB power spectrum
cls_file = '%s/%s' %(param_dict['data_folder'], param_dict['cls_file'])

if not pol:
    tqulen = 1
else:
    tqulen = 3
tqu_tit_arr = ['T', 'Q', 'U']


#sim stuffs
#total_sim_types = param_dict['total_sim_types'] #unlensed background and lensed clusters
total_clusters = param_dict['total_clusters']
total_randoms = param_dict['total_randoms'] #total_clusters * 10 #much more randoms to ensure we are not dominated by variance in background stack.

#cluster info
cluster_mass = param_dict['cluster_mass']
cluster_z = param_dict['cluster_z']

#cluster mass definitions
delta=param_dict['delta']
rho_def=param_dict['rho_def']
profile_name=param_dict['profile_name']

#cosmology
h=param_dict['h']
omega_m=param_dict['omega_m']
omega_lambda=param_dict['omega_lambda']
z_lss=param_dict['z_lss']
T_cmb=param_dict['T_cmb']

#cutouts specs 
cutout_size_am = param_dict['cutout_size_am'] #arcmins

#for estimating cmb gradient
apply_wiener_filter = param_dict['apply_wiener_filter']
lpf_gradient_filter = param_dict['lpf_gradient_filter']
cutout_size_am_for_grad = param_dict['cutout_size_am_for_grad'] #arcminutes
########################


########################
#get ra, dec or map-pixel grid
ra=np.linspace(x1,x2, nx) #arcmins
dec=np.linspace(x1,x2, nx) #arcmins
ra_grid, dec_grid=np.meshgrid(ra,dec)
########################

########################
#CMB power spectrum - read Cls now
if not pol:
    tqulen=1
else:
    tqulen=3
tqu_tit_arr=['T', 'Q', 'U']
el, cl = tools.get_cmb_cls(cls_file, pol = pol)
########################

########################
#get beam and noise
bl = tools.get_bl(beamval, el, make_2d = 1, mapparams = mapparams)
if ilc_file is None:
    nl_dic = tools.get_nl_dic(noiseval, el, pol = pol)
else:
    ilc_dic = np.load(ilc_file, allow_pickle = True).item()
    weights_arr, cl_residual_arr = ilc_dic['TT'][which_ilc]
    cl_residual_arr = np.interp(el, np.arange(len(cl_residual_arr)), cl_residual_arr)
    nl_dic = {}
    nl_dic['T'] = cl_residual_arr
print('\tkeys in nl_dict = %s' %(str(nl_dic.keys())))
########################

########################
#get foreground spectra if requested
if fg_gaussian:
    cl_fg_dic = tools.get_cl_fg(el = el, freq = 150, pol = pol)
########################

########################
#generating sims for model now
do_lensing=True
nclustersorrandoms=total_clusters
sim_type='clusters'
ra_grid_deg, dec_grid_deg = ra_grid/60., dec_grid/60.


#model_dic = {}
cluster_mass_arr = np.arange( minM, maxM+delM/10., delM ) * 1e14
cluster_z_arr = np.tile( cluster_z, len(cluster_mass_arr) )

for (cluster_mass, cluster_z) in zip(cluster_mass_arr, cluster_z_arr):


    keyname = (round(cluster_mass/1e14, 3), round(cluster_z, 3))
    print('\t###############')
    print('\tcreating model for %s' %(str(keyname)))
    print('\t###############')
    #model_dic[keyname] = {}

    np.random.seed( random_seed_for_models )
    print('\t\tsetting random seed for model generation. seed is %s' %(random_seed_for_models))

    M200c_list = np.tile(cluster_mass, total_clusters)
    redshift_list = np.tile(cluster_z, total_clusters)
    ra_list = dec_list = np.zeros(total_clusters)

    kappa_arr = lensing.get_convergence(ra_grid_deg, dec_grid_deg, ra_list, dec_list, M200c_list, redshift_list, param_dict)
    #print('\tShape of convergence array is %s' %(str(kappa_arr.shape)))

    sim_dic={}
    sim_dic[sim_type]={}
    sim_dic[sim_type]['sims'] = {}
    sim_dic[sim_type]['cmb_sims'] = {}
    print('\t\tcreating %s %s simulations' %(nclustersorrandoms, sim_type))
    for simcntr in range( totiters_for_model ):
        print('\t\t\tmodel dataset %s of %s' %(simcntr+1, totiters_for_model))
        cmb_sim_arr,sim_arr=[],[]
        #for i in tqdm(range(nclustersorrandoms)):
        for i in range(nclustersorrandoms):
            if not pol:
                cmb_map=np.asarray( [flatsky.make_gaussian_realisation(mapparams, el, cl[0])] )
                noise_map=np.asarray( [flatsky.make_gaussian_realisation(mapparams, el, nl_dic['T'])] )
                if fg_gaussian:
                    fg_map=np.asarray( [flatsky.make_gaussian_realisation(mapparams, el, cl_fg_dic['T'])] )
                else:
                    fg_map = np.zeros_like(noise_map)
            else:
                cmb_map=flatsky.make_gaussian_realisation(mapparams, el, cl[0], cl2=cl[1], cl12=cl[3], qu_or_eb='qu')
                noise_map_T=flatsky.make_gaussian_realisation(mapparams, el, nl_dic['T'])
                noise_map_Q=flatsky.make_gaussian_realisation(mapparams, el, nl_dic['P'])
                noise_map_U=flatsky.make_gaussian_realisation(mapparams, el, nl_dic['P'])
                noise_map=np.asarray( [noise_map_T, noise_map_Q, noise_map_U] )
                if fg_gaussian:
                    fg_map_T=flatsky.make_gaussian_realisation(mapparams, el, cl_fg_dic['T'])
                    fg_map_Q=flatsky.make_gaussian_realisation(mapparams, el, cl_fg_dic['P'])
                    fg_map_U=flatsky.make_gaussian_realisation(mapparams, el, cl_fg_dic['P'])
                    fg_map=np.asarray( [fg_map_T, fg_map_Q, fg_map_U] )
                else:
                    fg_map = np.zeros_like(noise_map)
            
            if do_lensing:
                cmb_map_unlensed = np.copy(cmb_map)
                cmb_map_lensed=[]
                for tqu in range(tqulen):
                    unlensed_cmb=np.copy( cmb_map[tqu] )
                    lensed_cmb=lensing.perform_lensing(ra_grid_deg, dec_grid_deg, unlensed_cmb, kappa_arr[i], mapparams)
                    if i == 0 and debug:
                        subplot(1,tqulen,tqu+1); imshow(lensed_cmb - unlensed_cmb, extent=[x1,x2,x1,x2], cmap=cmap); colorbar(); title(r'Sim=%s: %s' %(i, tqu_tit_arr[tqu])); 
                        axhline(lw=1.); axvline(lw=1.); xlabel(r'X [arcmins]'); 
                        if tqu == 0:
                            ylabel(r'Y [arcmins]')
                    cmb_map_lensed.append( lensed_cmb )
                if i == 0 and debug:
                    show()
                cmb_map=np.asarray(cmb_map_lensed)

            #add beam
            cmb_map = np.fft.ifft2( np.fft.fft2(cmb_map) * bl ).real
            fg_map = np.fft.ifft2( np.fft.fft2(fg_map) * bl ).real
                
            sim_map=np.copy(cmb_map)+noise_map+fg_map
            for tqu in range(tqulen):#mean subtraction for T(/Q/U)
                sim_map[tqu] -= np.mean(sim_map[tqu])
                cmb_map[tqu] -= np.mean(cmb_map[tqu])

            sim_arr.append( sim_map )
            cmb_sim_arr.append( cmb_map )
        sim_dic[sim_type]['sims'][simcntr]=np.asarray( sim_arr )
        sim_dic[sim_type]['cmb_sims'][simcntr]=np.asarray( cmb_sim_arr )

    #get gradient information for all cluster cutouts
    print('\t\tget gradient information for all cluster cutouts')
    for sim_type in sim_dic:
        sim_dic[sim_type]['cutouts_rotated'] = {}
        sim_dic[sim_type]['grad_mag'] = {}
        sim_dic[sim_type]['grad_orien'] = {}
        for simcntr in range( totiters_for_model ):
            print('\t\t\tmodel dataset %s of %s' %(simcntr+1, totiters_for_model))
            sim_arr=sim_dic[sim_type]['sims'][simcntr]
            cmb_sim_arr=sim_dic[sim_type]['cmb_sims'][simcntr]
            nclustersorrandoms=len(sim_arr)
            if apply_wiener_filter:
                if pol:
                    cl_signal_arr=[cl[0], cl[1], cl[1]]
                    cl_noise_arr=[nl_dic['T'], nl_dic['P'], nl_dic['P']]
                else:
                    cl_signal_arr=[cl[0]]
                    cl_noise_arr=[nl_dic['T']]

            #get median gradient direction and magnitude for all cluster cutouts + rotate them along median gradient direction.
            #for models, we will obtain gradient directions using cmb+noise+fg while we will simply rotate cmb sims
            grad_mag_arr, grad_orien_arr, cutouts_rotated_arr = tools.get_rotated_tqu_cutouts(cmb_sim_arr, sim_arr, nclustersorrandoms, tqulen, mapparams, cutout_size_am, perform_rotation = False, apply_wiener_filter=apply_wiener_filter, cl_signal = cl_signal_arr, cl_noise = cl_noise_arr, lpf_gradient_filter = lpf_gradient_filter, cutout_size_am_for_grad = cutout_size_am_for_grad)

            sim_dic[sim_type]['cutouts_rotated'][simcntr]=cutouts_rotated_arr
            sim_dic[sim_type]['grad_mag'][simcntr]=grad_mag_arr
            sim_dic[sim_type]['grad_orien'][simcntr]=grad_orien_arr

    #stack rotated cutouts + apply gradient magnitude weights
    print('\t\tstack rotated cutouts + apply gradient magnitude weights')
    model_dic = {}
    for sim_type in sim_dic:
        for simcntr in range( totiters_for_model ):
            model_dic[simcntr] = {}        
            print('\t\t\tmodel dataset %s of %s' %(simcntr+1, totiters_for_model))
            cutouts_rotated_arr=sim_dic[sim_type]['cutouts_rotated'][simcntr]
            grad_mag_arr=sim_dic[sim_type]['grad_mag'][simcntr]
            grad_orien_arr=sim_dic[sim_type]['grad_orien'][simcntr]

            stack=tools.stack_rotated_tqu_cutouts(cutouts_rotated_arr, weights_for_cutouts = grad_mag_arr)
            #print(weighted_stack.shape, weights.shape)
            model_dic[simcntr]['stack'] = stack
            model_dic[simcntr]['cutouts'] = [cutouts_rotated_arr, grad_mag_arr, grad_orien_arr]

            if (0):
                print(simcntr, sim_type, random_seed_for_models, keyname)
                modelbgfname = '../results/nx120_dx1/beam1.2/noise5/10amcutouts/nogaussianfg/T/models/clusters_700objects_1sims_randomseed100_mass0.000_z0.700.npy'
                modelbg = np.load(modelbgfname, allow_pickle = True).item()[0]
                subplot(131);imshow(stack[0]); colorbar(); 
                subplot(132);imshow(modelbg[0]); colorbar(); 
                curr_model = stack - modelbg
                subplot(133);imshow(curr_model[0]); colorbar(); show()
                sys.exit()

    fg_str=''
    if fg_gaussian:
        fg_str = 'withgaussianfg'
    else:
        fg_str = 'nogaussianfg'
    if add_cluster_tsz:
        fg_str = '%s_withclustertsz' %(fg_str)
    if add_cluster_ksz:
        fg_str = '%s_withclusterksz' %(fg_str)
    fg_str = fg_str.strip('_')
    mdef = 'm%s%s_%g' %(param_dict['delta'], param_dict['rho_def'], param_dict['cluster_mass'])
    op_folder = misc.get_op_folder(results_folder, nx, dx, beamval, noiseval, cutout_size_am, mdef=mdef, ilc_file=ilc_file, which_ilc = which_ilc, nclustersorrandoms = total_clusters, pol = pol, models = True, fg_str = fg_str)
    extrastr = '_randomseed%s_mass%.3f_z%.3f' %(random_seed_for_models, keyname[0], keyname[1])
    op_fname = misc.get_op_fname(op_folder, sim_type, nclustersorrandoms, totiters_for_model, extrastr = extrastr)
    np.save(op_fname, model_dic)
    logline = '\t\tResults dumped in %s' %(op_fname)
    print(logline)
sys.exit()
########################

########################
#save results
op_folder = '%s/nx%s_dx%s/beam%s/noise%s/%samcutouts/' %(results_folder, nx, dx, beamval, noiseval, cutout_size_am)
if pol:
    op_folder = '%s/TQU/' %(op_folder)
else:
    op_folder = '%s/T/' %(op_folder)
if not os.path.exists(op_folder): os.system('mkdir -p %s' %(op_folder))
op_fname = '%s/%s_%sobjects_%ssims%sto%s.npy' %(op_folder, sim_type, nclustersorrandoms, end-start, start, end)
sim_dic[sim_type].pop('sims')
np.save(op_fname, sim_dic)
logline = 'All completed. Results dumped in %s' %(op_fname)
print(logline)
########################


