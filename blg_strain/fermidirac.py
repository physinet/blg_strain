import numpy as np
from .utils.const import kB

def feq_func(E, EF, T=0):
    '''
    Fermi-Dirac distribution for calculating electron or hole occupation

    Arguments:
    - E: Energy (eV) - an array
    - EF: Fermi energy (eV)
    - T: Temperature (K)
    '''

    if T < 1e-10:
        T = 1e-10 # small but finite to avoid dividing by zero
    f = 1 / (1 + np.exp((E - EF) / (kB * T)))
    f[E<0] = 1 - f[E < 0] # for holes

    return f

def check_f_boundaries(f, thresh=0.01):
    '''
    Given an (n bands) x (Nkx kx points) x (Nky ky points) array of values for
    the Fermi-Dirac distribution, checks if the values are above a threshold
    along the boundaries of the k space spanned by kx and ky.
    '''
    assert f.ndim == 3 # n x Nkx x Nky

    for n in range(f.shape[0]): # loop over bands
        below_threshold = True # threshold to check if FD is small enough at boundaries of k-space
        for i in [0, -1]:
            if (f[n, i, :] > thresh).any():
                below_threshold = False
            elif (f[n, :, i] > thresh).any():
                below_threshold = False
        if not below_threshold:
            print('F-D dist in band %i not smaller than %f at boundaries!' %(n, thresh))
