import numpy as np, os

#################################################################################
#################################################################################
#################################################################################

def is_seq(o):
    """
    determine if the passed variable is an array.
    """
    return hasattr(o, '__len__')

################################################################################################################
################################################################################################################

def get_param_dict(paramfile):
    params, paramvals = np.genfromtxt(
        paramfile, delimiter = '=', unpack = True, autostrip = True, dtype='unicode')
    param_dict = {}
    for p,pval in zip(params,paramvals):
        if pval in ['T', 'True']:
            pval = True
        elif pval in ['F', 'False']:
            pval = False
        elif pval == 'None':
            pval = None
        else:
            try:
                pval = float(pval)
                if int(pval) == float(pval):
                    pval = int(pval)
            except:
                pass
        # replace unallowed characters in paramname
        p = p.replace('(','').replace(')','')
        param_dict[p] = pval
    return param_dict

################################################################################################################
################################################################################################################

def get_op_folder(results_folder, nx, dx, beamval, noiseval, cutout_size_am, pol = False):
    op_folder = '%s/nx%s_dx%s/beam%s/noise%s/%samcutouts/' %(results_folder, nx, dx, beamval, noiseval, cutout_size_am)
    if pol:
        op_folder = '%s/TQU/' %(op_folder)
    else:
        op_folder = '%s/T/' %(op_folder)
    if not os.path.exists(op_folder): os.system('mkdir -p %s' %(op_folder))
    return op_folder

################################################################################################################
################################################################################################################

def get_op_fname(op_folder, sim_type, nclustersorrandoms, end-start, start, end):
    return '%s/%s_%sobjects_%ssims%sto%s.npy' %(op_folder, sim_type, nclustersorrandoms, end-start, start, end)

################################################################################################################
################################################################################################################
