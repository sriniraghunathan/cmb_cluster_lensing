#!/usr/bin/env
########################

########################
#load desired modules
import numpy as np, sys, os, scipy as sc, argparse, glob, re
sys_path_folder='/Users/sraghunathan/Research/SPTPol/analysis/git/cmb_cluster_lensing/python/'
sys.path.append(sys_path_folder)

import flatsky, tools, lensing, foregrounds, misc

from pylab import *

import warnings
warnings.filterwarnings('ignore',category=RuntimeWarning)
print('\n')
########################

########################

fd_pref = '../results/nx120_dx1/beam1.2/noise5/10amcutouts/m500crit_2.3e+14/'
reqd_fd_arr = [
'500clusters/',
'2000clusters/',
'6300clusters/',
]

color_dict = {500: 'navy', 2000: 'darkgreen', 6300: 'orangered', 10000: 'darkred'} #set a colour based on the number of clusters

#get all folders - different Nclusters and w/o or w/ foregrounds
fd_arr = []
for curr_reqd_fd in reqd_fd_arr:
    fd = '%s/%s/' %(fd_pref, curr_reqd_fd)
    curr_fd_list = glob.glob('%s/*' %(fd))
    fd_arr.extend(curr_fd_list)

def get_pl_stuffs(fname):
    if fname.find('nogaussianfg')>-1:
        fg_str = 'No foregrounds'
        mkrval = 'o'
    else:
        fg_str = 'With foregrounds'
        mkrval = 'd'

    return fg_str, mkrval

def write_text(mean_snr, true_mass, total_clusters, fg_str, mdefstr = None, fsval = 14):
    ylocdelta = 0.1
    ###text(-2.3, rowcntr+ylocdelta-0.1, r'%s' %(expname), fontsize = fsval, color = colorval)
    #textval = r'$\overline{\rm SNR} = %.1f$ (%s = %.1f; N=%s)' %(mean_snr, mdefstr, true_mass, total_clusters)
    #text(1.6, rowcntr+ylocdelta, r'%s' %(textval), fontsize = fsval-2, color = colorval)
    textval = r'$\overline{\rm SNR} = %.1f$' %(mean_snr)
    text(0.8, rowcntr+ylocdelta, r'%s' %(textval), fontsize = fsval-1, color = colorval)
    text(-2.3, rowcntr-ylocdelta, r'\textsc{%s}' %(fg_str), fontsize = fsval, color = colorval)
    if mdefstr is not None:
        textval = r'%s = %.1f; N$_{\rm clus}$=%s' %(mdefstr, true_mass, total_clusters)
        text(0.8, rowcntr-ylocdelta-0.12, r'%s' %(textval), fontsize = fsval-1, color = colorval)

clf()
ax = subplot(111)
rowcntr = 0
msval = 8.
prev_total_clusters = 0
for fdcntr, fd in enumerate( fd_arr ):
    #searchstr = '%s/%s/results*' %(fd, estimator)
    searchstr = '%s/*/results*' %(fd)
    fname_arr = glob.glob(searchstr)#[0]
    for fname in fname_arr:
        res_dic = np.load(fname, allow_pickle = True).item()
        param_dict = res_dic['param_dict']

        total_clusters = param_dict['total_clusters']
        fg_str, mkrval = get_pl_stuffs(fname)
        colorval = color_dict[total_clusters]

        if (1): #for plotting 
            mdefstr = 'M$_{%s%s}$' %(param_dict['delta'], param_dict['rho_def'][0])
        true_mass = param_dict['cluster_mass']/1e14

        #-1 corresponds to combined likelihood; other key correspond to a respective sim number.
        likelihood_results = res_dic['likelihood']['T']

        master_loglarr = []
        for simcntr in likelihood_results:
            if simcntr == -1: #combined likelihood
                alphaval = 1.
                capsizeval = 3.
                continue
            else: #individual sims
                alphaval = 0.1
                capsizeval = 0.
            massarr, loglarr, massarr_mod, larr, recov_mass, snr = likelihood_results[simcntr]
            recov_mass, recov_mass_low_err, recov_mass_high_err = tools.get_width_from_sampling(massarr_mod, larr)
            recov_mass_err = (recov_mass_low_err + recov_mass_high_err)/2.
            #if abs(recov_mass - true_mass)/recov_mass_err>2.: continue
            master_loglarr.append(loglarr)
            #if simcntr == -1: write_text()
            errorbar( recov_mass - true_mass, rowcntr, xerr = recov_mass_err, color = colorval, marker = mkrval, ms = msval, alpha = alphaval, capsize = capsizeval)            

        combined_loglarr = np.sum(master_loglarr, axis = 0)
        massarr_mod, combined_larr, combined_recov_mass, combined_snr = tools.lnlike_to_like(massarr, combined_loglarr)
        combined_mean_mass, combined_mean_mass_low_err, combined_mean_mass_high_err = tools.get_width_from_sampling(massarr_mod, combined_larr)
        combined_mean_mass_err = (combined_mean_mass_low_err + combined_mean_mass_high_err)/2.
        mean_snr = combined_snr / np.sqrt(len(master_loglarr))
        errorbar( combined_mean_mass - true_mass, rowcntr, xerr = combined_mean_mass_err, color = colorval, marker = mkrval, ms = msval, alpha = alphaval, capsize = capsizeval)
        write_text(mean_snr, true_mass, total_clusters, fg_str, mdefstr = mdefstr, fsval=10)
        print(rowcntr, total_clusters, true_mass, combined_mean_mass, combined_mean_mass_err)
        rowcntr -= 1
        if total_clusters != prev_total_clusters:
            axhline(rowcntr-0.5, lw = 1., color = 'gray')
            prev_total_clusters = total_clusters
xlabel(r'M$_{\rm lens}$ - M$_{\rm true}$ [10$^{14}$ M$_{\odot}$]', fontsize = 14)
xlim(-2.4, 2.4)
#ylim(-1., rowcntr-1.)
ylim(rowcntr+0.5, 0.5)
axvline(lw = 0.5)
setp(ax.get_yticklabels(which = 'both'), visible=False)
ax.axes.yaxis.set_visible(False)
title(r'\textsc{CMB lensing: Temperature-only}', fontsize = 14)
#show(); sys.exit()
savefig('plots/sims_lensing_temp_500_2000_63000_10000.png', dpi = 200.)
show();
sys.exit()

