import numpy as np
from .utils.const import nu, eta0, eta3, eta4, \
                         gamma0, gamma1, gamma3, gamma4, \
                         dab, v0, v3, v4, hbar
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


sigmax = np.array([[0, 1], [1, 0]])
sigmay = np.array([[0, -1j], [1j, 0]])
sigmaz = np.array([[1, 0], [0, -1]])


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
    dabo = dab * o
    gamma1o = gamma1 * o

    # Gauge fields
    w3 = w(delta, idx=3, xi=xi, theta=0) * o
    w4 = w(delta, idx=4, xi=xi, theta=0) * o

    w3s = w3.conjugate()
    w4s = w4.conjugate()

    m = gamma1o / (2 * v0**2)

    ax = (-hbar**2 / (2 * m) * (Kx**2 - Ky**2) + xi * v3 * hbar*Kx + w3)
    ay =  - (hbar**2 / m * Kx * Ky + xi * v3 * hbar * Ky)
    az = Deltao / 2

    H = np.array([[az, ax - 1j * ay], [ax + 1j * ay, -az]])

    return H


def H2_dkx(Kx, Ky, xi=1):
    '''
    Returns the derivative (w.r.t. kx) of the 2x2 Hamiltonian
    Units are eV * m

    Parameters
    - Kx, Ky: Nkx x Nky array of wave vectors (nm^-1)
    - xi: valley index
    '''
    m = gamma1 / (2 * v0**2)

    ax = (-hbar ** 2 / (2 * m) * Kx + xi * v3 * hbar)
    ay = - hbar ** 2 / m * Ky
    az = 0 * Kx

    return np.array([[az, ax - 1j * ay], [ax + 1j * ay, -az]])


def H2_dky(Kx, Ky, xi=1):
    '''
    Returns the derivative (w.r.t. ky) of the 2x2 Hamiltonian
    Units are eV * m

    Parameters
    - Kx, Ky: Nkx x Nky array of wave vectors (nm^-1)
    - xi: valley index
    '''
    m = gamma1 / (2 * v0**2)

    ax = hbar ** 2 / (2 * m) * Ky
    ay = - (hbar ** 2 / m * Kx + xi * v3 * hbar)
    az = 0 * Kx

    return  np.array([[az, ax - 1j * ay], [ax + 1j * ay, -az]])
