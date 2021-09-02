import numpy as np, os, flatsky, foregrounds
import scipy as sc
import scipy.ndimage as ndimage

from pylab import *

#################################################################################
#################################################################################
#################################################################################

def get_cmb_cls(cls_file, pol=False):
    el, dl_tt, dl_ee, dl_bb, dl_te =np.loadtxt(cls_file, unpack=1)
    dl_all=np.asarray( [dl_tt, dl_ee, dl_bb, dl_te] )
    cl_all=dl_to_cl(el, dl_all)
    cl_tt, cl_ee, cl_bb, cl_te=cl_all #Cls in uK
    cl_dic={}
    cl_dic['TT'], cl_dic['EE'], cl_dic['BB'], cl_dic['TE']=cl_tt, cl_ee, cl_bb, cl_te
    if not pol:
        cl=[cl_tt]    
    else:
        cl=cl_all

    #loglog(el, cl_tt)
    #print(len(el))
    return el, cl

#################################################################################

def get_nl_dic(noiseval, el, pol=False):
    nl_dic={}
    if pol:
        nl=[]
        for n in noiseval:
            nl.append( get_nl(n, el) )
        nl=np.asarray( nl )
        nl_dic['T'], nl_dic['P']=nl[0], nl[1]
    else:
        nl=[get_nl(noiseval, el)]
        nl_dic['T']=nl[0]
    return nl_dic

#################################################################################

def get_cl_fg(el=None, freq=150, pol=False, units='uk', pol_frac_cib = 0.02, pol_frac_radio = 0.03, lmax=None):
    el_fg_tmp, cl_fg_tmp=foregrounds.get_foreground_power_spt('all', freq1=150, freq2=None, units=units, lmax=lmax)
    if el is None:
        el=np.copy(el_fg_tmp)
    cl_fg_temp=np.interp(el, el_fg_tmp, cl_fg_tmp)
    cl_fg_dic={}
    cl_fg_dic['T']=cl_fg_temp
    if pol:
        el_fg_tmp, cl_dg_cl=foregrounds.get_foreground_power_spt('DG-Cl', freq1=150, freq2=None, units='uk', lmax=None)
        el_fg, cl_dg_po=foregrounds.get_foreground_power_spt('DG-Po', freq1=150, freq2=None, units='uk', lmax=None)
        el_fg, cl_rg=foregrounds.get_foreground_power_spt('RG', freq1=150, freq2=None, units='uk', lmax=None)
        cl_dg=cl_dg_cl + cl_dg_po
        cl_dg=cl_dg * pol_frac_cib
        cl_rg=cl_rg * pol_frac_radio
        cl_fg_tmp=cl_dg + cl_rg
        cl_fg_pol=np.interp(el, el_fg_tmp, cl_fg_tmp)
        cl_fg_dic['P']=cl_fg_pol
    return cl_fg_dic

#################################################################################

def rotate_cutout(cutout, angle_in_deg):

    return ndimage.interpolation.rotate(cutout, angle_in_deg, reshape=False, mode='reflect')

#################################################################################

def extract_cutout(mapparams, cutout_size_am):

    ny, nx, dx=mapparams
    cutout_size=int( cutout_size_am/dx )
    ex1, ex2=int( (nx - cutout_size)/2 ), int( (nx+cutout_size)/2 )
    ey1, ey2=int( (ny - cutout_size)/2 ), int( (ny+cutout_size)/2 )

    return [ey1, ey2, ex1, ex2]

#################################################################################

def get_gradient(cutout, mapparams=None, apply_wiener_filter=True, cl_signal=None, cl_noise=None, lpf_gradient_filter=None, cutout_size_am_for_grad=6.):

    """
    determine the gradient magnitude and direction of the passed cutout
    """

    #get Wiener filter
    if apply_wiener_filter:
        assert mapparams is not None and cl_signal is not None and cl_noise is not None
        wiener_filter=flatsky.wiener_filter(mapparams, cl_signal, cl_noise)
    else:
        wiener_filter=np.ones( cutout.shape )

    #get LPF for CMB gradient estimation
    if lpf_gradient_filter:
        assert mapparams is not None
        lpf_gradient_filter=flatsky.get_lpf_hpf(mapparams, lpf_gradient_filter, filter_type=0)
    else:
        lpf_gradient_filter=np.ones( cutout.shape )

    #apply filter to cutout now
    cutout=np.fft.ifft2( np.fft.fft2( cutout ) * wiener_filter * lpf_gradient_filter ).real

    #extract desired portion of the cutout for gradient estimation
    if cutout_size_am_for_grad is not None:
        assert mapparams is not None
        ey1, ey2, ex1, ex2=extract_cutout(mapparams, cutout_size_am_for_grad)
        cutout=cutout[ey1:ey2, ex1:ex2]
        cutout-=np.mean(cutout)

    #get y and x gradients of the cutout
    cutout_grad=np.asarray( np.gradient(cutout) )

    #gradient magnitude
    grad_mag=np.hypot(cutout_grad[0], cutout_grad[1]) 

    #gradient direction
    grad_orientation=np.nan_to_num( np.degrees( np.arctan2 (cutout_grad[0],cutout_grad[1]) ) )

    return cutout_grad, grad_orientation, grad_mag

#################################################################################

def get_bl(beamval, el, make_2d=0, mapparams=None):

    """
    get Gaussian beam (1d or 2D)
    """

    # beamval can be a list or ndarray containing beam values
    if type(beamval) == np.ndarray or type(beamval) == list:
        bl=beamval
    else:
        fwhm_radians=np.radians(beamval/60.)
        sigma=fwhm_radians / np.sqrt(8. * np.log(2.))
        sigma2=sigma ** 2
        #bl=np.exp(el * (el+1) * sigma2)
        bl=np.exp(-0.5 * el * (el+1) * sigma2)

    #convert 1d Bl into 2D, if need be.
    if make_2d:
        assert mapparams is not None
        el=np.arange(len(bl))
        bl=flatsky.cl_to_cl2d(el, bl, mapparams) 

    return bl

################################################################################################################

def get_nl(noiseval, el, beamval=None, use_beam_window=False, uk_to_K=False, elknee_t=-1, alpha_knee=0):

    """
    get noise power spectrum (supports both white and 1/f atmospheric noise)
    """

    if uk_to_K: noiseval=noiseval/1e6

    if use_beam_window:
        bl=get_bl(beamval, el)

    delta_T_radians=noiseval * np.radians(1./60.)
    nl=np.tile(delta_T_radians**2., int(max(el)) + 1 )

    nl=np.asarray( [nl[int(l)] for l in el] )

    if use_beam_window: nl=nl/bl**2.

    if elknee_t != -1.:
        nl=np.copy(nl) * (1. + (elknee_t * 1./el)**alpha_knee )

    return nl

################################################################################################################
################################################################################################################
def dl_to_cl(el, cl_or_dl, inverse=0):
    dl_fac=(el * (el+1)/2./np.pi)
    if inverse:
        return cl_or_dl*dl_fac
    else:
        return cl_or_dl/dl_fac

################################################################################################################
################################################################################################################
#operations on cutouts

def get_rotated_tqu_cutouts_simple(sim_arr, grad_orientation_arr, totobjects, tqulen):

    cutouts_rotated_arr=[]
    for i in range(totobjects):
        tmp_cutouts_rotated=[]
        for tqu in range(tqulen):
            cutout=sim_arr[i][tqu]
            grad_orientation=grad_orientation_arr[i][tqu]            
            cutout_rotated=rotate_cutout( cutout, grad_orientation )
            cutout_rotated=cutout_rotated - np.mean(cutout_rotated)
            tmp_cutouts_rotated.append( cutout_rotated )

        cutouts_rotated_arr.append( np.asarray( tmp_cutouts_rotated ) )

    cutouts_rotated_arr=np.asarray(cutouts_rotated_arr)

    return cutouts_rotated_arr

def get_rotated_tqu_cutouts(sim_arr, sim_arr_for_grad_direction, totobjects, tqulen, mapparams, cutout_size_am, perform_rotation = True, apply_wiener_filter=True, cl_signal=None, cl_noise=None, lpf_gradient_filter=None, cutout_size_am_for_grad=6.):

    """
    get median gradient direction and magnitude for all cluster cutouts + rotate them along median gradient direction.
    """

    ey1, ey2, ex1, ex2=extract_cutout(mapparams, cutout_size_am)
    
    cutouts_rotated_arr=[]
    grad_mag_arr=[]
    grad_orien_arr=[]
    #for i in tqdm(range(totobjects)):
    for i in range(totobjects):
        tmp_grad_mag_arr=[]
        tmp_cutouts_rotated=[]
        tmp_grad_orien_arr=[]
        for tqu in range(tqulen):
            cutout_grad, grad_orientation, grad_mag=get_gradient(sim_arr_for_grad_direction[i][tqu], mapparams=mapparams, apply_wiener_filter=apply_wiener_filter, cl_signal=cl_signal[tqu], cl_noise=cl_noise[tqu], lpf_gradient_filter=lpf_gradient_filter, cutout_size_am_for_grad=cutout_size_am_for_grad)

            cutout=sim_arr[i][tqu][ey1:ey2, ex1:ex2]
            median_grad_mag = np.median(grad_mag)
            median_grad_orientation = np.median(grad_orientation)
            if (0):
                median_grad_orientation = round(median_grad_orientation, 1)
                median_grad_mag = round(median_grad_mag, 1)

            if perform_rotation:
                cutout_rotated=rotate_cutout( cutout, median_grad_orientation )
            else:
                cutout_rotated = np.copy( cutout )
            cutout_rotated=cutout_rotated - np.mean(cutout_rotated)

            '''
            subplot(131);imshow(sim_arr_for_grad_direction[i][tqu][ey1:ey2, ex1:ex2]); colorbar();
            subplot(132);imshow(sim_arr[i][tqu][ey1:ey2, ex1:ex2]); colorbar();
            subplot(133);imshow(cutout_rotated); colorbar(); show(); sys.exit()
            '''

            tmp_cutouts_rotated.append( cutout_rotated )
            tmp_grad_mag_arr.append( median_grad_mag )
            tmp_grad_orien_arr.append( median_grad_orientation )

        grad_mag_arr.append( np.asarray(tmp_grad_mag_arr) )
        cutouts_rotated_arr.append( np.asarray( tmp_cutouts_rotated ) )
        grad_orien_arr.append( np.asarray( tmp_grad_orien_arr ) )

    grad_mag_arr=np.asarray(grad_mag_arr)
    grad_orien_arr=np.asarray(grad_orien_arr)
    cutouts_rotated_arr=np.asarray(cutouts_rotated_arr)

    return grad_mag_arr, grad_orien_arr, cutouts_rotated_arr

def stack_rotated_tqu_cutouts(cutouts, weights_for_cutouts=None, perform_random_rotation = False):
    if weights_for_cutouts is None:
        weights_for_cutouts=np.ones_like(cutouts)

    if perform_random_rotation:
        for i in range(len(cutouts)):
            tqulen = len(cutouts[i])
            for tqu in range(tqulen):
                cutouts[i][tqu] = rotate_cutout(cutouts[i][tqu], np.random.random() * 360.)

    weighted_stack=np.sum( cutouts[:, :] * weights_for_cutouts[:, :, None, None], axis=0)
    weights=np.sum( weights_for_cutouts, axis=0)
    stack=weighted_stack / weights[:, None, None]

    return stack

################################################################################################################
################################################################################################################
def perform_simple_jackknife_sampling(total, howmany_jk_samples):
    each_split_should_contain=int(total * 1./howmany_jk_samples)
    fullarr=np.arange(total)
    inds_to_pick=np.arange(len(fullarr))
    already_picked_inds=[]
    jk_samples=[]
    for n in range(howmany_jk_samples):
        inds=np.random.choice(inds_to_pick, size=each_split_should_contain, replace=0)
        inds_to_delete=np.where (np.in1d(inds_to_pick, inds) == True)[0]
        inds_to_pick=np.delete(inds_to_pick, inds_to_delete)
        #push all on the non inds dic into - because for each JK we will ignore the files for this respective sim
        tmp=np.in1d(fullarr, inds)
        non_inds=np.where(tmp == False)[0]
        jk_samples.append( (non_inds) )
    return np.asarray( jk_samples )

def get_jk_covariance(cutouts, howmany_jk_samples, weights=None, only_T=False):

    total_clusters=len(cutouts)
    jk_samples=perform_simple_jackknife_sampling(total_clusters, howmany_jk_samples)

    simarr=np.arange(howmany_jk_samples)
    ny, nx=cutouts[0][0].shape
    npixels=ny * nx
    if only_T:
        tqu_len=1
    else:
        tqu_len=len(cutouts[0])

    if weights is None:
        weights=np.ones( total_clusters )

    stacked_cutouts_for_jk_cov=np.zeros( (tqu_len * npixels, howmany_jk_samples) )

    for jkcnt, n in enumerate( simarr ):

        #print('JK=%s of %s' %(jkcnt, howmany_jk_samples), end=' ')
        non_inds=jk_samples[n]

        tqu_cluster_stack=[]
        for tqucntr in range(tqu_len):

            weighted_cluster_stack_arr=[]
            curr_cutouts, curr_weights=cutouts[non_inds, tqucntr], weights[non_inds, tqucntr]
            for (c, w) in zip( curr_cutouts, curr_weights ):

                weighted_cluster_stack_arr.append( c * w )

            weighted_cluster_stack_arr=np.asarray(weighted_cluster_stack_arr)

            cluster_stack=np.sum( weighted_cluster_stack_arr, axis=0)/np.sum( curr_weights )
            #imshow(cluster_stack, interpolation='bicubic', cmap=cm.RdYlBu); colorbar(); title(n); show(); sys.exit()
            tqu_cluster_stack.append(cluster_stack.flatten())
        stacked_cutouts_for_jk_cov[:, n]=np.asarray(tqu_cluster_stack).flatten()
        #subplot(5,5,n+1); imshow(stacked_cutouts_for_jk_cov[:, n].reshape(ny,nx), interpolation='bicubic', cmap=cm.RdYlBu); colorbar(); title(n); 

    #show(); #sys.exit()
    mean=np.mean(stacked_cutouts_for_jk_cov, axis=1)
    for jkcnt, n in enumerate( simarr ):
        stacked_cutouts_for_jk_cov[:, n]=stacked_cutouts_for_jk_cov[:, n] - mean

    #print(stacked_cutouts_for_jk_cov.shape)
    jk_cov=(howmany_jk_samples - 1) * np.cov(stacked_cutouts_for_jk_cov)    


    return jk_cov

################################################################################################################
################################################################################################################
def downsample_map(data, N=2): #from N.Whitehorn
    from numpy import average, split
    ''' original from N.WHitehorn
    width=data.shape[0]
    height= data.shape[1]
    return average(split(average(split(data, width // N, axis=1), axis=-1), height // N, axis=1), axis=-1)
    '''
    height, width=data.shape
    return average(split(average(split(data, width // N, axis=1), axis=-1), height // N, axis=1), axis=-1)

def get_lnlikelihood(data, model, cov):

    """
    function to calculate the likelihood given data, model, covariance matrix
    """

    cov=np.mat(cov)
    cov_inv=sc.linalg.pinv2(cov)

    sign, logdetval=np.linalg.slogdet(cov)
    logdetval=logdetval * sign

    d=data.flatten()## - np.mean(MAP.flatten())
    m=model.flatten()## - np.mean(MODEL.flatten())
    d=d-m

    logLval= -0.5 * np.asarray( np.dot(d.T, np.dot( cov_inv, d ))).squeeze()

    return logLval

def fitting_func_gaussian(p, p0, X, DATA=None, return_fit=0):
    import scipy.special
    fitfunc=lambda p, x: p[1]*(np.exp(-(x-p[2])**2/(2*p[3]**2)))
    if not return_fit:
        return fitfunc(p, X) - DATA
    else:
        return fitfunc(p, X)

def likelihood_finer_resol(M, L, intrp_type=2):

    import scipy.optimize as optimize

    deltaM=np.diff(M)[0]
    M_ip=np.arange(min(M),max(M),deltaM/100.)

    if intrp_type == 2: #Guassian fitting
        #first guess a good parameter
        Mfit=M[np.argmax(L)]
        gau_width=abs(Mfit - M[np.argmin(abs(L))])#/2.35 * 2.
        p0=[0.,np.max(L),Mfit,gau_width]
        p1, success=optimize.leastsq(fitting_func_gaussian, p0, args=(p0, M, L))

        L_ip=fitting_func_gaussian(p1, p1, M_ip, return_fit=1)

    return M_ip, L_ip

def lnlike_to_like(M, lnlike, intrp_type=0):

    lnlike=lnlike - max(lnlike)

    '''
    if intrp_type == 1:
        deltaM=np.diff(M)[0]
        M_ip=np.arange(min(M),max(M),deltaM/100.)
        lnlike_ip=np.interp(M_ip, M, lnlike)
        M=np.copy(M_ip)
        lnlike=np.copy(lnlike_ip)
    '''

    delta_chisq=max(lnlike) - lnlike[0]
    snr=np.sqrt(2 * delta_chisq)

    L=np.exp(lnlike); L/=max(L)
    recov_mass=M[np.argmax(L)]

    if intrp_type <= 1: #no interpolation
        return M, L, recov_mass, snr

    M_ip, L_ip=likelihood_finer_resol(M, L, intrp_type=intrp_type)
    L_ip /= max(L_ip)
    recov_mass=M_ip[np.argmax(L_ip)]

    return M_ip, L_ip, recov_mass, snr

def random_sampler(x, y, howmanysamples = 100000, burn_in = 5000):
    import scipy.integrate as integrate
    import scipy.interpolate as interpolate

    norm = integrate.simps(y, x) #area under curve for norm
    y = y/norm #normalise dn/dM here

    cdf = np.asarray([integrate.simps(y[:i+1], x[:i+1]) for i in range(len(x))])
    cdf_inv = interpolate.interp1d(cdf, x)

    random_sample = cdf_inv(np.random.rand(howmanysamples))

    return random_sample[burn_in:]  

def get_width_from_sampling(x, likelihood_curve):#, sigma_value = [1.]):
    randoms = random_sampler(x, likelihood_curve)
    mean_mass = x[np.argmax(likelihood_curve)]
    low_err = mean_mass - np.percentile(randoms, 16.)
    high_err = np.percentile(randoms, 84.) - mean_mass

    return mean_mass, low_err, high_err

