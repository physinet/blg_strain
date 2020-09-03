# Calculation of microscopic quantities from the bands

import numpy as np
from .utils.const import kB, hbar, hbar_J, m_e, a, q
from scipy.interpolate import RectBivariateSpline
from scipy.integrate import simps

def feq_func(E, EF, T=0):
    '''
    Fermi-Dirac distribution for equilibrium electron occupation feq. To
    determine hole occupation, take -(1-feq) (such that carrier density is
    negative for hole doping).

    Arguments:
    - E: Energy (eV) - an array with arbitrary dimensions
    - EF: Fermi level (eV)
    - T: Temperature (K)
    '''

    with np.errstate(divide='ignore', over='ignore'):
        f = 1 / (1 + np.exp((E - EF) / (kB * T)))

    return f


def f_relaxation(kxa, kya, splE, EF, T, Efield, tau, N):
    '''
    Solution of the Boltzmann equation under the relaxation time approximation
    for an equilibrium occupation that follows the Fermi-Dirac distribution.
    The occupation is distorted by an applied electric field.

    Arguments:
    - kxa, kya: Nkx, Nky arrays of kxa, kya points
    - splE: spline for energy for given band (eV)
    - EF: Fermi level (eV)
    - T: Temperature (K)
    - Efield: an electric field in the x direction (V/m)
    - tau: relaxation time (s)
    - N: number of points to use in integration
    '''
    qxa = np.linspace(kxa.min(), kxa.max(), N)

    Qxa, Kxa, Kya = np.meshgrid(qxa, kxa, kya, indexing='ij')
    for i in range(Kxa.shape[1]):
        # make an array ranging from far left of integration window to each Kx
        Qxa[:, i] = np.linspace(kxa.min()-1e-10, Kxa[0, i], N)

    f0 = feq_func(splE(Qxa, Kya, grid=False), EF, T=T)
    integrand = np.exp(-hbar_J * (Kxa - Qxa) / (tau * Efield * q * a)) \
                * f0 / (tau * Efield * q)

    return simps(integrand, Qxa, axis=0)


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
