import numpy as np
from .utils.const import hbar, gamma1, a0


def H_4x4(Kxa, Kya, sl, Delta=0):
    '''
    Calculates the 4x4 low-energy Hamiltonian for uniaxially strained BLG.

    Parameters:
    - Kxa, Kya: Nkx x Nky array of wave vectors
    - sl: instance of the StrainedLattice class
    - Delta: interlayer asymmetry (eV)

    Returns:
    - H: Hamiltonian array shape 4 x 4 x Nkx x Nky
    '''
    # K vector
    Ka = np.stack([Kxa, Kya], axis=-1)

    o = np.zeros_like(Kxa, dtype='complex128') # Array to give shape to const

    # Nearest-neighbor matrix elements
    H0 = o.copy()
    H3 = o.copy()
    H4 = o.copy()
    for (delta, gamma0, gamma3, gamma4) in zip(sl.deltas, sl.gamma0s,
                                               sl.gamma3s, sl.gamma4s):
        H0 += -gamma0 * np.exp(1j * Ka.dot(delta))
        H3 += gamma3 * np.exp(-1j * Ka.dot(delta))
        H4 += gamma4 * np.exp(1j * Ka.dot(delta))

    # Next-nearest neighbor matrix element
    Hn = o.copy()
    for (deltan, gamman) in zip(sl.deltans, sl.gammans):
        Hn += -gamman * np.exp(1j * Ka.dot(deltan))

    H = np.array([
        [-Delta / 2 + Hn, H3, H4, H0],
        [H3.conj(), Delta/2 + Hn, H0.conj(), H4.conj()],
        [H4.conj(), H0, Delta/2 + Hn + sl.DeltaAB, gamma1 + o],
        [H0.conj(), H4, gamma1 + o, -Delta/2 + Hn + sl.DeltaAB]
    ]) # Using "o" gives elements proper shape

    return H


def dH_4x4(Kxa, Kya, sl):
    '''
    Returns the gradient of the 4 x 4 Hamiltonian
    Units are eV * m

    Parameters
    - Kxa, Kya: Nkx x Nky array of wave vectors
    - sl: instance of the StrainedLattice class

    Returns:
    H_dkx, H_dky: derivatives of H, shape 4 x 4 x Nkx x Nky
    '''
    # K vector
    Ka = np.stack([Kxa, Kya], axis=-1)

    o = np.zeros_like(Kxa, dtype='complex128') # Array to give shape to const

    # Nearest-neighbor matrix elements
    dH0x = o.copy()
    dH0y = o.copy()
    dH3x = o.copy()
    dH3y = o.copy()
    dH4x = o.copy()
    dH4y = o.copy()
    for (delta, gamma0, gamma3, gamma4) in zip(sl.deltas, sl.gamma0s,
                                               sl.gamma3s, sl.gamma4s):
        dH0x += -gamma0 * np.exp(1j * Ka.dot(delta)) * 1j * delta[0]
        dH0y += -gamma0 * np.exp(1j * Ka.dot(delta)) * 1j * delta[1]
        dH3x += -gamma3 * np.exp(-1j * Ka.dot(delta)) * 1j * delta[0]
        dH3y += -gamma3 * np.exp(-1j * Ka.dot(delta)) * 1j * delta[1]
        dH4x += gamma4 * np.exp(1j * Ka.dot(delta)) * 1j * delta[0]
        dH4y += gamma4 * np.exp(1j * Ka.dot(delta)) * 1j * delta[1]

    # Next-nearest neighbor matrix element
    dHnx = o.copy()
    dHny = o.copy()
    for (deltan, gamman) in zip(sl.deltans, sl.gammans):
        dHnx += -gamman * np.exp(1j * Ka.dot(deltan)) * 1j * deltan[0]
        dHny += -gamman * np.exp(1j * Ka.dot(deltan)) * 1j * deltan[1]

    # Multiply by a0 to make the derivative w.r.t. kx or ky, not kx*a0
    H_dkx = a0 * np.array([
        [dHnx, dH3x, dH4x, dH0x],
        [dH3x.conj(), dHnx, dH0x.conj(), dH4x.conj()],
        [dH4x.conj(), dH0x, dHnx, o],
        [dH0x.conj(), dH4x, o, dHnx]
    ]) # Using "o" gives elements proper shape
    H_dky = a0 * np.array([
        [dHny, dH3y, dH4y, dH0y],
        [dH3y.conj(), dHny, dH0y.conj(), dH4y.conj()],
        [dH4y.conj(), dH0y, dHny, o],
        [dH0y.conj(), dH4y, o, dHny]
    ]) # Using "o" gives elements proper shape

    return H_dkx, H_dky
