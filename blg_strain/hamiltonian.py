import numpy as np
from .utils.const import nu, eta0, eta3, eta4, \
                         gamma0, gamma1, gamma3, gamma4, \
                         dab, v0, v3, v4, hbar, meff
from .utils.params import w

def Hfunc(Kx, Ky, xi=1, Delta=0, delta=0, theta=0, twobytwo=False):
    '''
    Calculates the low-energy Hamiltonian for uniaxially strained BLG.

    Parameters:
    - Kx, Ky: Nkx x Nky array of wave vectors (nm^-1)
    - xi: valley index (+1 for K, -1 for K')
    - Delta: interlayer asymmetry (eV)
    - delta: uniaxial strain
    - theta: angle of uniaxial strain to zigzag axis
    - twobytwo: if True, use 2x2 Hamiltonian (N=2). If False, use 4x4 (N=4)

    Returns:
    - H: Hamiltonian array shape N x N x Nkx x Nky
    '''
    if twobytwo:
        return H_2by2(Kx, Ky, xi=xi, Delta=Delta, delta=delta, theta=theta)
    else:
        return H_4by4(Kx, Ky, xi=xi, Delta=Delta, delta=delta, theta=theta)


def H_4by4(Kx, Ky, xi=1, Delta=0, delta=0, theta=0):
    '''
    Calculates the 4x4 low-energy Hamiltonian for uniaxially strained BLG.

    Parameters:
    - Kx, Ky: Nkx x Nky array of wave vectors (nm^-1)
    - xi: valley index (+1 for K, -1 for K')
    - Delta: interlayer asymmetry (eV)
    - delta: uniaxial strain
    - theta: angle of uniaxial strain to zigzag axis

    Returns:
    - H: Hamiltonian array shape 4 x 4 x Nkx x Nky
    '''

    # Array to give proper shape to constants
    o = np.ones_like(Kx)
    Deltao = Delta * o
    dabo = dab * o
    gamma1o = gamma1 * o

    # Gauge fields
    w3 = w(delta, idx=3, xi=xi, theta=0) * o
    w4 = w(delta, idx=4, xi=xi, theta=0) * o

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


def H_2by2(Kx, Ky, xi=1, Delta=0, delta=0, theta=0):
    '''
    Calculates the 2x2 low-energy Hamiltonian for uniaxially strained BLG.

    Parameters:
    - Kx, Ky: Nkx x Nky array of wave vectors (nm^-1)
    - xi: valley index (+1 for K, -1 for K')
    - Delta: interlayer asymmetry (eV)
    - delta: uniaxial strain
    - theta: angle of uniaxial strain to zigzag axis

    Returns:
    - H: Hamiltonian array shape 2 x 2 x Nkx x Nky
    '''
    # Array to give proper shape to constants
    o = np.ones_like(Kx)
    Deltao = Delta * o

    # Gauge fields
    w3 = w(delta, idx=3, xi=xi, theta=0) * o
    w3s = w3.conjugate()

    # Momentum
    px, py = hbar * Kx, hbar * Ky
    pi = xi * px + 1j * py
    pidag = xi * px - 1j * py

    H = np.array([
        [-Deltao / 2, -pidag ** 2 / (2 * meff) + xi * v3 * pi + w3],
        [-pi **2 / (2 * meff) + xi * v3 * pidag + w3s,   Deltao / 2]
    ])

    return H


def H2_dkx(Kx, Ky, xi=1):
    '''
    Returns the derivative (w.r.t. kx) of the 2x2 Hamiltonian
    Units are eV * m

    Parameters
    - Kx, Ky: Nkx x Nky array of wave vectors (nm^-1)
    - xi: valley index
    '''
    # Momentum
    px, py = hbar * Kx, hbar * Ky
    pi = xi * px + 1j * py
    pidag = xi * px - 1j * py

    # dH/dkx = (dH/dpi)*(dpi/dkx) + (dHdpidag)*(dpidag/dkx)
    #        = (dH/dpi)*hbar + (dH/dpidag)*hbar
    return np.array([
        [0 * Kx, -pidag / meff + xi * v3],
        [-pi / meff + xi * v3,  0 * Kx]
    ]) * hbar


def H2_dky(Kx, Ky, xi=1):
    '''
    Returns the derivative (w.r.t. ky) of the 2x2 Hamiltonian
    Units are eV * m

    Parameters
    - Kx, Ky: Nkx x Nky array of wave vectors (nm^-1)
    - xi: valley index
    '''
    # Momentum
    px, py = hbar * Kx, hbar * Ky
    pi = xi * px + 1j * py
    pidag = xi * px - 1j * py

    # dH/dky = (dH/dpi)*(dpi/dky) + (dHdpidag)*(dpidag/dky)
    #        = (dH/dpi)*i*hbar + (dH/dpidag)*(-i)*hbar
    return np.array([
        [0 * Kx, pidag / meff + xi * v3],
        [-pi / meff - xi * v3,  0 * Kx]
    ]) * 1j * hbar
