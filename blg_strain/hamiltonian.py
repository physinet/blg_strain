import numpy as np
from .utils.const import nu, eta0, eta3, eta4, \
                         gamma0, gamma1, gamma3, gamma4, \
                         dab, v0, v3, v4, hbar

def Hfunc(Kx, Ky, xi=1, Delta=0, delta=0, theta=0):
    '''
    Calculates the 4x4 low-energy Hamiltonian for uniaxially strained BLG.

    Parameters:
    - Kx, Ky: Nky x Nkx array of wave vectors (nm^-1)
    - xi: valley index (+1 for K, -1 for K')
    - Delta: interlayer asymmetry (eV)
    - delta: uniaxial strain
    - theta: angle of uniaxial strain to zigzag axis

    Returns:
    - H: Hamiltonian array shape 4 x 4 x Nky x Nkx
    '''

    # Array to give proper shape to constants
    o = np.ones_like(Kx)
    Deltao = Delta * o
    dabo = dab * o
    gamma1o = gamma1 * o

    # Gauge fields
    w3 = 3 / 4 * np.exp(-1j*2*xi*theta)*(1+nu)*delta*(eta3 - eta0)*gamma3*o
    w4 = 3 / 4 * np.exp(-1j*2*xi*theta)*(1+nu)*delta*(eta4 - eta0)*gamma4*o

    w3s = w3.conjugate()
    w4s = w4.conjugate()

    # Momentum
    px, py = hbar * Kx, hbar * Ky
    pi = xi * px + 1j * py
    pidag = xi * px - 1j * py

    H = np.array([
        [-1/2 * Deltao, v3 * pi + w3, -v4 * pidag - w4s, v0*pidag],
        [v3 * pidag + w3s,  1/2 * Deltao, v0 * pi, -v4 * pi - w4],
        [-v4 * pi - w4, v0 * pidag, 1/2 * Deltao + dabo, gamma1o],
        [v0 * pi, -v4 * pidag - w4s, gamma1o, -1/2 * Deltao + dabo]
    ])

    return H

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
