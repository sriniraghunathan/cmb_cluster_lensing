#!/usr/bin/env python
########################

########################
#load desired modules
import numpy as np, sys, os, scipy as sc, argparse
sys_path_folder='../python/'
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
parser.add_argument('-start', dest='start', action='store', help='start', type=int, default=0)
parser.add_argument('-end', dest='end', action='store', help='end', type=int, default=10)
parser.add_argument('-paramfile', dest='paramfile', action='store', help='paramfile', type=str, required=True)#='params.ini')
parser.add_argument('-clusters_or_randoms', dest='clusters_or_randoms', action='store', help='clusters_or_randoms', type=str, default='clusters')
parser.add_argument('-random_seed_for_sims', dest='random_seed_for_sims', action='store', help='random_seed_for_sims', type=int, default=-1)#111)

args = parser.parse_args()
args_keys = args.__dict__
for kargs in args_keys:
    param_value = args_keys[kargs]

    if isinstance(param_value, str):
        cmd = '%s = "%s"' %(kargs, param_value)
    else:
        cmd = '%s = %s' %(kargs, param_value)
    exec(cmd)

if clusters_or_randoms == 'randoms':
    start, end = 0, 1

########################

########################
print('\tread/get necessary params')
param_dict = misc.get_param_dict(paramfile)

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

try:
    pol_frac_radio = param_dict['pol_frac_radio']
except:
    pol_frac_radio = False

try:
    pol_frac_cib = param_dict['pol_frac_cib']
except:
    pol_frac_cib = False

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
    cl_fg_dic = tools.get_cl_fg(el = el, freq = 150, pol = pol, pol_frac_cib = pol_frac_cib, pol_frac_radio = pol_frac_radio)
########################

########################
#plot
if debug:
    ax =subplot(111, yscale='log', xscale='log')
    plot(el, cl[0], color='black', label=r'TT')
    plot(el, nl_dic['T'], color='black', ls ='--', label=r'Noise: T')
    if fg_gaussian:
        plot(el, cl_fg_dic['T'], color='black', ls ='-.', label=r'Foregrounds: T')
    if pol:
        plot(el, cl[1], color='orangered', label=r'EE')
        plot(el, nl_dic['P'], color='orangered', ls ='--', label=r'Noise: P')
        if fg_gaussian:
            plot(el, cl_fg_dic['P'], color='orangered', ls ='-.', label=r'Foregrounds: P')
    legend(loc=1)
    ylim(1e-10, 1e4)
    ylabel(r'$C_{\ell}$ [$\mu$K$^{2}$]')
    xlabel(r'Multipole $\ell$')
    show()
########################

########################
if clusters_or_randoms == 'clusters':
    print('\tgetting NFW convergence for lensing')
    #NFW lensing convergence
    ra_grid_deg, dec_grid_deg = ra_grid/60., dec_grid/60.

    M200c_list = np.tile(cluster_mass, total_clusters)
    redshift_list = np.tile(cluster_z, total_clusters)
    ra_list = dec_list = np.zeros(total_clusters)

    kappa_arr = lensing.get_convergence(ra_grid_deg, dec_grid_deg, ra_list, dec_list, M200c_list, redshift_list, param_dict)
    print('\tShape of convergence array is %s' %(str(kappa_arr.shape)))
    #imshow(kappa_arr[0]); colorbar(); show(); sys.exit()
########################

########################
#generating sims
sim_dic={}
if clusters_or_randoms == 'clusters': #cluster lensed sims
    do_lensing=True
    nclustersorrandoms=total_clusters
    sim_type='clusters'
elif clusters_or_randoms == 'randoms':
    do_lensing=False        
    nclustersorrandoms=total_randoms        
    sim_type='randoms'
sim_dic[sim_type]={}
sim_dic[sim_type]['sims'] = {}
print('\tcreating %s %s simulations' %(nclustersorrandoms, sim_type))
for simcntr in range( start, end ):

    print('\t\tmock dataset %s of %s' %(simcntr+1, end-start))

    ########################
    #pick different mdpl2 tsz/ksz for each iteration of the sim run
    if sim_type == 'clusters' and (add_cluster_ksz or add_cluster_tsz):
        print('\t\t\tgetting mdpl2 tsz/ksz for cluster correlated foregrounds')
        mdpl2_dic, mdpl2_cutout_size_am = foregrounds.get_mdpl2_cluster_tsz_ksz(total_clusters, dx, return_tsz = add_cluster_tsz, return_ksz = add_cluster_ksz)
        if add_cluster_ksz: mdpl2_ksz_cutouts = mdpl2_dic['ksz']
        if add_cluster_tsz: mdpl2_tsz_cutouts = mdpl2_dic['tsz']
    ########################

    if random_seed_for_sims != -1:
        randomseedval = random_seed_for_sims * simcntr
        np.random.seed(randomseedval)

    sim_arr=[]
    for i in tqdm(range(nclustersorrandoms)):
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

        #add cluster correalted ksz/tsz from MDPL2 if requested for temperature
        for tqu in range(tqulen):
            if sim_type == 'clusters' and tqu == 0:
                if add_cluster_ksz: #ksz
                    ey1, ey2, ex1, ex2=tools.extract_cutout(mapparams, mdpl2_cutout_size_am)
                    fg_map[tqu, ey1:ey2, ex1:ex2]+=mdpl2_ksz_cutouts[i]        
                if add_cluster_tsz: #tsz
                    ey1, ey2, ex1, ex2=tools.extract_cutout(mapparams, mdpl2_cutout_size_am)
                    fg_map[tqu, ey1:ey2, ex1:ex2]+=mdpl2_tsz_cutouts[i]

        #add beam
        cmb_map = np.fft.ifft2( np.fft.fft2(cmb_map) * bl ).real
        fg_map = np.fft.ifft2( np.fft.fft2(fg_map) * bl ).real
        
        sim_map=cmb_map + noise_map + fg_map

        for tqu in range(tqulen):#mean subtraction for T(/Q/U)
            sim_map[tqu] -= np.mean(sim_map[tqu])
        sim_arr.append( sim_map )
    sim_dic[sim_type]['sims'][simcntr]=np.asarray( sim_arr )
########################


########################
if debug: #just plot the last map
    close()
    clf()

    if not pol:
        figure(figsize=(6,4))
        subplots_adjust(hspace=0.2, wspace=0.1)
        subplot(141);imshow(cmb_map[0], extent=[x1,x2,x1,x2], cmap=cmap); colorbar(); title(r'CMB')
        axhline(lw=1.); axvline(lw=1.); xlabel(r'X [arcmins]'); ylabel(r'Y [arcmins]')
        subplot(142);imshow(noise_map[0], extent=[x1,x2,x1,x2], cmap=cmap); colorbar(); title(r'Noise')
        axhline(lw=1.); axvline(lw=1.); xlabel(r'X [arcmins]'); #ylabel(r'Y [arcmins]')
        subplot(143);imshow(fg_map[0], extent=[x1,x2,x1,x2], cmap=cmap); colorbar(); title(r'Foregrounds')
        axhline(lw=1.); axvline(lw=1.); xlabel(r'X [arcmins]'); #ylabel(r'Y [arcmins]')
        subplot(144);imshow(sim_map[0], extent=[x1,x2,x1,x2], cmap=cmap); colorbar(); title(r'CMB + Noise')
        axhline(lw=1.); axvline(lw=1.); xlabel(r'X [arcmins]'); #ylabel(r'Y [arcmins]')
    else:
        figure(figsize=(12,12))
        subplots_adjust(hspace=0.3, wspace=0.1)
        for tqucntr in range(tqulen):
            tmp_sim_cntr=4 #cmb, noise, fg, total
            subplot(tqulen,4,(tqucntr*tmp_sim_cntr)+1);imshow(cmb_map[tqucntr], extent=[x1,x2,x1,x2], cmap=cmap); colorbar(); title(r'CMB: %s' %(tqu_tit_arr[tqucntr]))
            axhline(lw=1.); axvline(lw=1.); ylabel(r'Y [arcmins]')
            subplot(tqulen,4,(tqucntr*tmp_sim_cntr)+2);imshow(noise_map[tqucntr], extent=[x1,x2,x1,x2], cmap=cmap); colorbar(); title(r'Noise: %s' %(tqu_tit_arr[tqucntr]))
            axhline(lw=1.); axvline(lw=1.); #xlabel(r'X [arcmins]'); #ylabel(r'Y [arcmins]')
            subplot(tqulen,4,(tqucntr*tmp_sim_cntr)+3);imshow(fg_map[tqucntr], extent=[x1,x2,x1,x2], cmap=cmap); colorbar(); title(r'Foregrounds: %s' %(tqu_tit_arr[tqucntr]))
            axhline(lw=1.); axvline(lw=1.); xlabel(r'X [arcmins]'); #ylabel(r'Y [arcmins]')
            subplot(tqulen,4,(tqucntr*tmp_sim_cntr)+4);imshow(sim_map[tqucntr], extent=[x1,x2,x1,x2], cmap=cmap); colorbar(); title(r'CMB + Noise: %s' %(tqu_tit_arr[tqucntr]))
            axhline(lw=1.); axvline(lw=1.); xlabel(r'X [arcmins]'); #ylabel(r'Y [arcmins]')
    show()


if debug: #get power spectrum of maps to ensure sims are fine
    ax=subplot(111, xscale='log', yscale='log')
    colorarr=['black', 'orangered', 'darkgreen']
    colorarr_noise=['gray', 'crimson', 'lightgreen']
    colorarr_fg=['k', 'red', 'green']
    if pol:
        cl_theory=[cl[0], cl[1]/2, cl[1]/2]
    else:
        cl_theory=[cl[0]]
    for tqucntr in range(tqulen):
        curr_el, curr_cl=flatsky.map2cl(mapparams, cmb_map[tqucntr], filter_2d=bl**2.)
        curr_el, curr_nl=flatsky.map2cl(mapparams, noise_map[tqucntr])
        curr_el, curr_cl_fg=flatsky.map2cl(mapparams, fg_map[tqucntr])

        plot(el, cl_theory[tqucntr], color=colorarr[tqucntr])#, label=r'CMB theory')
        plot(curr_el, curr_cl, color=colorarr[tqucntr], ls ='--')#, label=r'CMB map')
        if tqucntr == 0: 
            TP = 'T' 
        else: 
            TP = 'P'
        plot(el, nl_dic[TP], color=colorarr_noise[tqucntr])#, label=r'Noise theory')
        plot(curr_el, curr_nl, color=colorarr_noise[tqucntr], ls ='--')#, label=r'Noise map')
        if fg_gaussian:
            plot(el, cl_fg_dic[TP], color=colorarr_fg[tqucntr])
            plot(curr_el, curr_cl_fg, color=colorarr_fg[tqucntr], ls ='--')

    ylabel(r'$C_{\ell}$ [$\mu$K$^{2}$]')
    xlabel(r'Multipole $\ell$')
    show()

########################

########################
#get gradient information for all cluster cutouts
print('\tget gradient information for all cluster cutouts')
for sim_type in sim_dic:
    sim_dic[sim_type]['cutouts_rotated'] = {}
    sim_dic[sim_type]['grad_mag'] = {}
    for simcntr in range( start, end ):
        print('\t\tmock dataset %s of %s' %(simcntr+1, end-start))
        sim_arr=sim_dic[sim_type]['sims'][simcntr]
        nclustersorrandoms=len(sim_arr)
        if apply_wiener_filter:
            if pol:
                cl_signal_arr=[cl[0], cl[1]/2., cl[1]/2.]
                cl_noise_arr=[nl_dic['T'], nl_dic['P'], nl_dic['P']]
            else:
                cl_signal_arr=[cl[0]]
                cl_noise_arr=[nl_dic['T']]

        #get median gradient direction and magnitude for all cluster cutouts + rotate them along median gradient direction.
        grad_mag_arr, grad_orien_arr, cutouts_rotated_arr = tools.get_rotated_tqu_cutouts(sim_arr, sim_arr, nclustersorrandoms, tqulen, mapparams, cutout_size_am, apply_wiener_filter=apply_wiener_filter, cl_signal = cl_signal_arr, cl_noise = cl_noise_arr, lpf_gradient_filter = lpf_gradient_filter, cutout_size_am_for_grad = cutout_size_am_for_grad)
        
        sim_dic[sim_type]['cutouts_rotated'][simcntr]=cutouts_rotated_arr
        sim_dic[sim_type]['grad_mag'][simcntr]=grad_mag_arr    

########################


########################
#stack rotated cutouts + apply gradient magnitude weights
print('\tstack rotated cutouts + apply gradient magnitude weights')
sim_dic[sim_type]['stack'] = {}
for sim_type in sim_dic:
    for simcntr in range( start, end ):
        print('\t\tmock dataset %s of %s' %(simcntr+1, end-start))
        cutouts_rotated_arr=sim_dic[sim_type]['cutouts_rotated'][simcntr]
        grad_mag_arr=sim_dic[sim_type]['grad_mag'][simcntr]

        stack = tools.stack_rotated_tqu_cutouts(cutouts_rotated_arr, weights_for_cutouts = grad_mag_arr)
        sim_dic[sim_type]['stack'][simcntr]=stack
########################

########################
#save results
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
op_folder = misc.get_op_folder(results_folder, nx, dx, beamval, noiseval, cutout_size_am, mdef = mdef, ilc_file = ilc_file, which_ilc = which_ilc, nclustersorrandoms = total_clusters, pol = pol, fg_str = fg_str)
op_fname = misc.get_op_fname(op_folder, sim_type, nclustersorrandoms, end-start, start, end, random_seed_for_sims = random_seed_for_sims)
sim_dic[sim_type].pop('sims')
if clusters_or_randoms == 'randoms':
    sim_dic[sim_type].pop('cutouts_rotated')
    sim_dic[sim_type].pop('grad_mag')
sim_dic['param_dict']=param_dict
np.save(op_fname, sim_dic)
logline = 'All completed. Results dumped in %s' %(op_fname)
print(logline)
########################

########################
if debug: #plot results
    dummysimcntr = 0
    ex1, ex2=-cutout_size_am/2. * dx, cutout_size_am/2. * dx
    cluster_stack=sim_dic['clusters']['stack'][dummysimcntr]
    if 'randoms' in sim_dic:
        random_stack=sim_dic['randoms']['stack'][dummysimcntr]
    else:
        random_stack = np.zeros_like(cluster_stack)
    final_stack=cluster_stack - random_stack
    sbpl=1
    tr, tc=tqulen, 3

    close()
    clf()
    figure(figsize=(10,10))
    subplots_adjust(hspace=0.2, wspace=0.1)
    fsval=10
    intrp_val='bicubic'

    def add_labels(ax, sbpl):
        axhline(lw=1.); axvline(lw=1.)
        if sbpl>=7:
            xlabel(r'X [arcmins]')
        if (sbpl-1)%tc == 0: 
            ylabel(r'Y [arcmins]')
        return ax

    for tqu in range(tqulen):
        ax=subplot(tr, tc, sbpl); imshow(cluster_stack[tqu], cmap=cmap, extent=[ex1, ex2, ex1, ex2], interpolation=intrp_val); colorbar(); title(r'Cluster stack: %s' %(tqu_tit_arr[tqu]), fontsize=fsval);
        ax=add_labels(ax, sbpl); sbpl+=1
        subplot(tr, tc, sbpl); imshow(random_stack[tqu], cmap=cmap, extent=[ex1, ex2, ex1, ex2], interpolation=intrp_val); colorbar(); title(r'Random stack: %s' %(tqu_tit_arr[tqu]), fontsize=fsval);axhline(lw=1.); axvline(lw=1.)
        ax=add_labels(ax, sbpl); sbpl+=1
        subplot(tr, tc, sbpl); imshow(final_stack[tqu], cmap=cmap, extent=[ex1, ex2, ex1, ex2], interpolation=intrp_val); colorbar(); 
        title(r'Lensing stack: %s (%s clusters)' %(tqu_tit_arr[tqu], total_clusters), fontsize=fsval);axhline(lw=1.); axvline(lw=1.)
        ax=add_labels(ax, sbpl); sbpl+=1
    show()
########################
sys.exit()

########################
#get JK based covariance from cluster cuouts

#BG subtracted rotated cluster cuouts
dummysimcntr = 0
cluster_cutouts_rotated_arr=sim_dic['clusters']['cutouts_rotated'][dummysimcntr]# - random_stack
if (0):
    tmp_stack=np.mean(cluster_cutouts_rotated_arr, axis=0)
    for tqu in range(len(tmp_stack)):
        subplot(1, 3, tqu+1); imshow(tmp_stack[tqu]); colorbar()
    show(); sys.exit()
cluster_grad_mag_arr=sim_dic['clusters']['grad_mag'][dummysimcntr]

jk_cov=tools.get_jk_covariance(cluster_cutouts_rotated_arr, param_dict['howmany_jk_samples'], weights=cluster_grad_mag_arr, only_T=True)
#print(jk_cov.shape)
clf()
imshow(jk_cov, cmap=cmap); colorbar(); show()
########################




