import numpy as np, os, flatsky
import scipy as sc
import scipy.ndimage as ndimage

from pylab import *

#################################################################################
#################################################################################
#################################################################################

def is_seq(o):
    """
    determine if the passed variable is an array.
    """
    return hasattr(o, '__len__')

#################################################################################

def rotate_cutout(cutout, angle_in_deg):

    return ndimage.interpolation.rotate(cutout, angle_in_deg, reshape = False, mode = 'reflect')

#################################################################################

def extract_cutout(mapparams, cutout_size_am):

    ny, nx, dx = mapparams
    cutout_size = int( cutout_size_am/dx )
    ex1, ex2 = int( (nx - cutout_size)/2 ), int( (nx+cutout_size)/2 )
    ey1, ey2 = int( (ny - cutout_size)/2 ), int( (ny+cutout_size)/2 )

    return [ey1, ey2, ex1, ex2]

#################################################################################

def get_gradient(cutout, mapparams = None, apply_wiener_filter = True, cl_signal = None, cl_noise = None, lpf_gradient_filter = None, cutout_size_am_for_grad = 6.):

    """
    determine the gradient magnitude and direction of the passed cutout
    """

    #get Wiener filter
    if apply_wiener_filter:
        assert mapparams is not None and cl_signal is not None and cl_noise is not None
        wiener_filter = flatsky.wiener_filter(mapparams, cl_signal, cl_noise)
    else:
        wiener_filter = np.ones( cutout.shape )

    #get LPF for CMB gradient estimation
    if lpf_gradient_filter:
        assert mapparams is not None
        lpf_gradient_filter = flatsky.get_lpf_hpf(mapparams, lpf_gradient_filter, filter_type = 0)
    else:
        lpf_gradient_filter = np.ones( cutout.shape )

    #apply filter to cutout now
    cutout = np.fft.ifft2( np.fft.fft2( cutout ) * wiener_filter * lpf_gradient_filter ).real

    #extract desired portion of the cutout for gradient estimation
    if cutout_size_am_for_grad is not None:
        assert mapparams is not None
        ey1, ey2, ex1, ex2 = extract_cutout(mapparams, cutout_size_am_for_grad)
        cutout = cutout[ey1:ey2, ex1:ex2]
        cutout-=np.mean(cutout)

    #get y and x gradients of the cutout
    cutout_grad = np.asarray( np.gradient(cutout) )

    #gradient magnitude
    grad_mag = np.hypot(cutout_grad[0], cutout_grad[1]) 

    #gradient direction
    grad_orientation = np.nan_to_num( np.degrees( np.arctan2 (cutout_grad[0],cutout_grad[1]) ) )

    return cutout_grad, grad_orientation, grad_mag

#################################################################################

def get_bl(beamval, el, make_2d = 0, mapparams = None):

    """
    get Gaussian beam (1d or 2D)
    """

    # beamval can be a list or ndarray containing beam values
    if type(beamval) == np.ndarray or type(beamval) == list:
        bl = beamval
    else:
        fwhm_radians = np.radians(beamval/60.)
        sigma = fwhm_radians / np.sqrt(8. * np.log(2.))
        sigma2 = sigma ** 2
        #bl = np.exp(el * (el+1) * sigma2)
        bl = np.exp(-0.5 * el * (el+1) * sigma2)

    #convert 1d Bl into 2D, if need be.
    if make_2d:
        assert mapparams is not None
        el = np.arange(len(bl))
        bl = flatsky.cl_to_cl2d(el, bl, mapparams) 

    return bl

################################################################################################################

def get_nl(noiseval, el, beamval = None, use_beam_window = False, uk_to_K = False, elknee_t = -1, alpha_knee = 0):

    """
    get noise power spectrum (supports both white and 1/f atmospheric noise)
    """

    if uk_to_K: noiseval = noiseval/1e6

    if use_beam_window:
        bl = get_bl(beamval, el)

    delta_T_radians = noiseval * np.radians(1./60.)
    nl = np.tile(delta_T_radians**2., int(max(el)) + 1 )

    nl = np.asarray( [nl[int(l)] for l in el] )

    if use_beam_window: nl = nl/bl**2.

    if elknee_t != -1.:
        nl = np.copy(nl) * (1. + (elknee_t * 1./el)**alpha_knee )

    return nl

################################################################################################################
################################################################################################################

def perform_simple_jackknife_sampling(total, howmany_jk_samples):
    each_split_should_contain = int(total * 1./howmany_jk_samples)
    fullarr = np.arange(total)
    inds_to_pick = np.arange(len(fullarr))
    already_picked_inds = []
    jk_samples = []
    for n in range(howmany_jk_samples):
        inds = np.random.choice(inds_to_pick, size = each_split_should_contain, replace = 0)
        inds_to_delete = np.where (np.in1d(inds_to_pick, inds) == True)[0]
        inds_to_pick = np.delete(inds_to_pick, inds_to_delete)
        #push all on the non inds dic into - because for each JK we will ignore the files for this respective sim
        tmp = np.in1d(fullarr, inds)
        non_inds = np.where(tmp == False)[0]
        jk_samples.append( (non_inds) )
    return np.asarray( jk_samples )


def get_jk_covariance(cutouts, howmany_jk_samples, weights = None, only_T = False):

    total_clusters = len(cutouts)
    jk_samples = perform_simple_jackknife_sampling(total_clusters, howmany_jk_samples)

    simarr = np.arange(howmany_jk_samples)
    ny, nx = cutouts[0][0].shape
    npixels = ny * nx
    if only_T:
        tqu_len = 1
    else:
        tqu_len = len(cutouts[0])

    if weights is None:
        weights = np.ones( total_clusters )

    stacked_cutouts_for_jk_cov = np.zeros( (tqu_len * npixels, howmany_jk_samples) )

    for jkcnt, n in enumerate( simarr ):

        #print('JK = %s of %s' %(jkcnt, howmany_jk_samples), end = ' ')
        non_inds = jk_samples[n]

        tqu_cluster_stack = []
        for tqucntr in range(tqu_len):

            weighted_cluster_stack_arr = []
            curr_cutouts, curr_weights = cutouts[non_inds, tqucntr], weights[non_inds, tqucntr]
            for (c, w) in zip( curr_cutouts, curr_weights ):

                weighted_cluster_stack_arr.append( c * w)

            weighted_cluster_stack_arr = np.asarray(weighted_cluster_stack_arr)

            cluster_stack = np.sum( weighted_cluster_stack_arr, axis = 0)/np.sum( curr_weights )
            #imshow(cluster_stack, interpolation = 'bicubic', cmap = cm.RdYlBu); colorbar(); title(n); show(); sys.exit()
            tqu_cluster_stack.append(cluster_stack.flatten())
        stacked_cutouts_for_jk_cov[:, n] = np.asarray(tqu_cluster_stack).flatten()
        #subplot(5,5,n+1); imshow(stacked_cutouts_for_jk_cov[:, n].reshape(ny,nx), interpolation = 'bicubic', cmap = cm.RdYlBu); colorbar(); title(n); 

    #show(); #sys.exit()
    '''
    mean = np.mean(stacked_cutouts_for_jk_cov, axis = 1)
    for jkcnt, n in enumerate( simarr ):
        stacked_cutouts_for_jk_cov[:, n] = stacked_cutouts_for_jk_cov[:, n] - mean
    '''

    jk_cov = (howmany_jk_samples - 1) * np.cov(stacked_cutouts_for_jk_cov)    


    return jk_cov

################################################################################################################

