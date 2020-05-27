# Calculation of microscopic quantities from the bands

import numpy as np
from .utils.const import kB, hbar, m_e

def meff_func(kx, ky, E):
    '''
    Calculates effective mass tensor from the curvature of the bands.
    Specifically, this returns the inverse of the reciprocal effective mass
    tensor, with components (1/m)_{ij} = 1/hbar^2  (d^2E)/(dk_i dk_j)

    The tensor is in units of the electron mass m_e.

    Parameters:
    - kx, ky: Nkx, Nky arrays of kx, ky points
    - E: N(=4) x Nkx x Nky array of energy eigenvalues

    Returns:
    - meff: N(=4) x Nkx x Nky x 2 x 2 array.
        The 1st dimension indexes the energy bands
        The 2nd/3rd dimensions index over ky and kx
        The 4th/5th dimensions are the 2x2 effective mass tensor
    '''
    E_dkx, E_dky = np.gradient(E, kx, ky, axis=(1,2), edge_order=2) # axis1 = y
                                                                    # axis2 = x

    E_dkx_dkx, E_dkx_dky = np.gradient(E_dkx, kx, ky, axis=(1,2), edge_order=2)
    E_dky_dkx, E_dky_dky = np.gradient(E_dky, kx, ky, axis=(1,2), edge_order=2)

    if E.shape[0] != 4:
        raise Exception('Something is wrong... size of E is not 4 x ...')

    oneoverm = np.zeros((E.shape[0], len(kx), len(ky), 2, 2))

    oneoverm[:, :, :, 0, 0] = E_dkx_dkx / hbar**2
    oneoverm[:, :, :, 0, 1] = E_dky_dkx / hbar**2
    oneoverm[:, :, :, 1, 0] = E_dkx_dky / hbar**2
    oneoverm[:, :, :, 1, 1] = E_dky_dky / hbar**2

    # np.linalg.inv will operate over last two axes
    return np.linalg.inv(oneoverm) / m_e  # m_e definition takes care of eV -> J


def feq_func(E, EF, T=0):
    '''
    Fermi-Dirac distribution for calculating electron or hole occupation

    Arguments:
    - E: Energy (eV) - an array with arbitrary dimensions
    - EF: Fermi energy (eV)
    - T: Temperature (K)
    '''

    if T < 1e-10:
        T = 1e-10 # small but finite to avoid dividing by zero
    f = 1 / (1 + np.exp((E - EF) / (kB * T)))
    f[E<0] = -(1 - f[E < 0])  # hole occupation is 1-f, and we give it a (-)
                              # so they contribute as (-) to carrier density

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
    - E: Energy (eV) - an ... x Nkx x Nky array
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
