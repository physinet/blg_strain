# Calculation of microscopic quantities from the bands

import numpy as np
from .utils.const import kB, hbar, m_e
from scipy.interpolate import RectBivariateSpline

def feq_func(E, EF, T=0):
    '''
    Fermi-Dirac distribution for equilibrium electron occupation feq. To
    determine hole occupation, take -(1-feq) (such that carrier density is
    negative for hole doping).

    Arguments:
    - E: Energy (eV) - an array with arbitrary dimensions
    - EF: Fermi energy (eV)
    - T: Temperature (K)
    '''

    with np.errstate(divide='ignore', over='ignore'):
        f = 1 / (1 + np.exp((E - EF) / (kB * T)))

    return f


def check_f_boundaries(f, thresh=0.01):
    '''
    Given an N(=4) x Nkx x Nky array of values for
    the Fermi-Dirac distribution, checks if the values are above a threshold
    along the boundaries of the k space spanned by kx and ky.
    Prints a warning if this condition is not met.
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


def grad_feq_func(kx, ky, E, EF, T=0):
    '''
    Gradient of Fermi-Dirac distribution (calculated using gradient of feq_func)

    Because grad_feq is sharply peaked, I would not expect this to perform well.

    Arguments:
    - kx, ky: Nkx, Nky arrays of kx, ky points
    - E: Energy (eV) - a ... x Nkx x Nky array
    - EF: Fermi energy (eV)
    - T: Temperature (K)

    Returns:
    - f_dkx, f_dky - components of gradient of feq (each with same shape as E)
    '''
    f = feq_func(E, EF, T)
    f_dkx, f_dky = np.gradient(f, kx, ky, axis=(-2,-1))

    return f_dkx, f_dky


def grad_feq_func_2(kx, ky, E, EF, T=0):
    '''
    Gradient of Fermi-Dirac distribution (calculated using gradient of energy)

    Arguments:
    - kx, ky: Nkx, Nky arrays of kx, ky points
    - E: Energy (eV) - an ... x Nkx x Nky array
    - EF: Fermi energy (eV)
    - T: Temperature (K)

    Returns:
    - f_dkx, f_dky - components of gradient of feq (each with same shape as E)
    '''
    if T < 1e-2:
        T = 1e-2 # small but finite to avoid dividing by zero
    E_dkx, E_dky = np.gradient(E, kx, ky, axis=(-2, -1))

    sech2 = np.cosh((E - EF) / (2 * kB * T)) ** -2

    return [-EE/(4 * kB * T) * sech2 for EE in (E_dkx, E_dky)]
