import numpy as np
from .utils.const import nu, eta0, eta3, eta4, \
                         gamma0, gamma1, gamma3, gamma4, \
                         dab, v0, v3, v4, hbar

def Hfunc(kx, ky, xi=1, Delta=0, delta=0, theta=0):
    '''
    Calculates the 4x4 low-energy Hamiltonian for uniaxially strained BLG.
    Compatible at this point only with single values of kx and ky
    (array implementation to come)

    Parameters:
    - kx, ky: Wave vectors (nm^-1)
    - xi: valley index (+1 for K, -1 for K')
    - Delta: interlayer asymmetry (eV)
    - delta: uniaxial strain
    - theta: angle of uniaxial strain to zigzag axis
    '''
    deltap = -nu * delta  # strain along transverse direction

    # Gauge fields
    w3 = 3 / 4 * np.exp(-1j*2*xi*theta)*(delta-deltap)*(eta3 - eta0)*gamma3
    w4 = 3 / 4 * np.exp(-1j*2*xi*theta)*(delta-deltap)*(eta4 - eta0)*gamma4

    w3s = w3.conjugate()
    w4s = w4.conjugate()

    px, py = hbar * kx, hbar * ky
    pi = xi * px + 1j * py
    pidag = xi * px - 1j * py

    return np.array([
        [-1/2 * Delta,      v3 * pi + w3,       -v4 * pidag - w4s,  v0*pidag],
        [v3 * pidag + w3s,  1/2 * Delta,        v0 * pi,         4 * pi - w4],
        [-v4 * pi - w4,     v0 * pidag,         1/2 * Delta + dab,    gamma1],
        [v0 * pi,           -v4 * pidag - w4s,  gamma1,     -1/2 * Delta + dab]
    ])

def H_dkx(xi=1):
    '''
    Returns the 4x4 derivative (w.r.t. kx) of the Hamiltonian
    Units are eV * m

    Parameters
    - xi: valley index (+1 for K, -1 for K')
    '''
    return xi * hbar * np.array([
        [0, v3, -v4, v0],
        [v3, 0, v0, -v4],
        [-v4, v0, 0, 0],
        [v0, -v4, 0, 0]
    ])

def H_dky(xi=1):
    '''
    Returns the 4x4 derivative (w.r.t. ky) of the Hamiltonian
    Units are eV * m

    Parameters
    - xi: valley index (unused)
    '''
    return 1j * hbar * np.array([
        [0, v3, v4, -v0],
        [-v3, 0, v0, -v4],
        [-v4, -v0, 0, 0],
        [v0, v4, 0, 0]
    ])
