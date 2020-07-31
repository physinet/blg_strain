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
        H3 += -gamma3 * np.exp(1j * Ka.dot(delta))
        H4 += -gamma4 * np.exp(1j * Ka.dot(delta))

    # Next-nearest neighbor matrix element
    Hn = o.copy()
    for (deltan, gamman) in zip(sl.deltans, sl.gammans):
        Hn += -gamman * np.exp(1j * Ka.dot(deltan))

    H = np.array([
        [-Delta / 2 + Hn, H3, -H4.conj(), H0.conj()],
        [H3.conj(), Delta/2 + Hn, H0, -H4],
        [-H4, H0.conj(), Delta/2 + Hn + sl.DeltaAB, gamma1 + o],
        [H0, -H4.conj(), gamma1 + o, -Delta/2 + Hn + sl.DeltaAB]
    ]) # Using "o" gives elements proper shape

    return H


def dH_4x4(Kxa, Kya, sl, xy=0):
    '''
    Returns the derivative (w.r.t. kx or ky) of the 4 x 4 Hamiltonian
    Units are eV

    Parameters
    - Kxa, Kya: Nkx x Nky array of wave vectors
    - sl: instance of the StrainedLattice class
    - xy: if 0 (1) returns the x (y) derivative

    Returns:
    H_dkx, H_dky: derivatives of H, shape 4 x 4 x Nkx x Nky
    '''
    # K vector
    Ka = np.stack([Kxa, Kya], axis=-1)

    o = np.zeros_like(Kxa, dtype='complex128') # Array to give shape to const

    # Nearest-neighbor matrix elements
    dH0 = o.copy()
    dH3 = o.copy()
    dH4 = o.copy()
    for (delta, gamma0, gamma3, gamma4) in zip(sl.deltas, sl.gamma0s,
                                               sl.gamma3s, sl.gamma4s):
        dH0 += -gamma0 * np.exp(1j * Ka.dot(delta)) * 1j * delta[xy]
        dH3 += -gamma3 * np.exp(1j * Ka.dot(delta)) * 1j * delta[xy]
        dH4 += -gamma4 * np.exp(1j * Ka.dot(delta)) * 1j * delta[xy]

    # Next-nearest neighbor matrix element
    dHn = o.copy()
    for (deltan, gamman) in zip(sl.deltans, sl.gammans):
        dHn += -gamman * np.exp(1j * Ka.dot(deltan)) * 1j * deltan[xy]

    # Multiply by a0 to make the derivative w.r.t. kx or ky, not kx*a0
    return a0 * np.array([
        [dHn, dH3, -dH4.conj(), dH0.conj()],
        [dH3.conj(), dHn, dH0, -dH4],
        [-dH4, dH0.conj(), dHn, o],
        [dH0, -dH4.conj(), o, dHn]
    ]) # Using "o" gives elements proper shape
