import numpy as np
from .utils.const import nu, eta0, eta3, eta4, \
                         gamma0, gamma1, gamma3, gamma4, \
                         dab, v0, v3, v4, hbar, meff
from .utils.params import w


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
    w3 = w(delta, idx=3, xi=xi, theta=theta) * o
    w4 = w(delta, idx=4, xi=xi, theta=theta) * o

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
    pi = px + 1j * py  # note: no xi!
    pidag = px - 1j * py

    # Moulsdale with added trigonal warping terms.
    H = np.array([
        [-Deltao / 2, -pidag ** 2 / (2 * meff) + v3 * pi + w3],
        [-pi ** 2 / (2 * meff) + v3 * pidag + w3s, Deltao / 2]
    ])

    # Battilomo
    # note this is written in a different basis than above (or interlayer
    # displacement is defined oppositely)
    # H = np.array([
    #     [Deltao / 2, -1/(2*meff) * (px ** 2 - py ** 2) + xi * v3 * px  + w3 + 1j/meff * px * py + 1j * xi * v3 * py],
    #     [-1/(2*meff) * (px ** 2 - py ** 2) + xi * v3 * px  + w3s - 1j/meff * px * py - 1j * xi * v3 * py,   -Deltao / 2]
    # ])

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
    pi = xi *  px + 1j * py
    pidag = xi * px - 1j * py

    # dH/dkx = (dH/dpi)*(dpi/dkx) + (dH/dpidag)*(dpidag/dkx)
    #        = (dH/dpi)*xi*hbar + (dH/dpidag)*xi*hbar
    return np.array([
        [0 * Kx, -pidag / meff + v3],
        [-pi / meff + v3,  0 * Kx]
    ]) * xi * hbar


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

    # dH/dky = (dH/dpi)*(dpi/dky) + (dH/dpidag)*(dpidag/dky)
    #        = (dH/dpi)*i*hbar + (dH/dpidag)*(-i)*hbar
    return np.array([
        [0 * Kx, pidag / meff +  v3],
        [-pi / meff - v3,  0 * Kx]
    ]) * 1j * hbar
