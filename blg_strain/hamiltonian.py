import numpy as np
from .utils.const import hbar, gamma1


def H_4by4(Kxa, Kya, Delta=0, sl=None):
    '''
    Calculates the 4x4 low-energy Hamiltonian for uniaxially strained BLG.

    Parameters:
    - Kxa, Kya: Nkx x Nky array of wave vectors
    - Delta: interlayer asymmetry (eV)
    - sl: instance of the StrainedLattice class

    Returns:
    - H: Hamiltonian array shape 4 x 4 x Nkx x Nky
    '''
    # K vector
    Ka = np.stack([Kxa, Kya], axis=-1)

    o = np.zeros_like(Kxa, dtype='complex128') # Array to give  shape to const
    I = np.eye(2) # 2x2 identity matrix

    # Nearest-neighbor matrix elements
    H0 = o.copy()
    H3 = o.copy()
    H4 = o.copy()
    for (delta, gamma0, gamma3, gamma4) in zip(sl.deltas, sl.gamma0s,
                                               sl.gamma3s, sl.gamma4s):
        H0 += gamma0 * np.exp(1j * Ka.dot(delta))
        H3 += gamma3 * np.exp(1j * Ka.dot(delta))
        H4 += gamma4 * np.exp(1j * Ka.dot(delta))

    # Next-nearest neighbor matrix element
    Hn = o.copy()
    for (deltan, gamman) in zip(sl.deltans, sl.gammans):
        Hn += gamman * np.exp(1j * Ka.dot(deltan))

    H = np.array([
        [-Delta / 2 + Hn, H3, -H4.conj(), H0.conj()],
        [H3.conj(), Delta/2 + Hn, H0, -H4],
        [-H4, H0.conj(), Delta/2 + Hn + sl.DeltaAB, gamma1 + o],
        [H0, -H4.conj(), gamma1 + o, -Delta/2 + Hn + sl.DeltaAB]
    ]) # Adding "o" to gamma1 and Hn to Delta/2 gives elements proper shape

    return H


def dH_4by4(Kxa, Kya, eps=0, theta=0):
    '''
    Returns the derivative (w.r.t. kx*a) of the 4 x 4 Hamiltonian
    Units are eV

    Parameters
    - Kxa, Kya: Nkx x Nky array of wave vectors
    - eps: uniaxial strain
    - theta: angle of uniaxial strain to zigzag axis

    Returns:
    H_dkx, H_dky: derivatives of H, shape 4 x 4 x Nkx x Nky
    '''
    raise Exception('Incorporate strained lattice classs')
    # K vector
    Ka = np.stack([Kxa, Kya], axis=-1)

    o = np.zeros_like(Kxa, dtype='complex128') # Array to give  shape to const
    I = np.eye(2) # 2x2 identity matrix

    # strain tensor
    strain = strain_tensor(eps, theta)

    # Transform bonds under strain
    deltas_p = [(I+strain).dot(delta) for delta in deltas]
    deltans_p = [(I+strain).dot(deltan) for deltan in deltans]

    # Nearest neighbor elements
    gammas = [c.gamma0, c.gamma3, c.gamma4]
    etas = [eta0, eta3, eta4]
    dHdxs = np.array([o, o, o])
    dHdys = np.array([o, o, o])
    for delta, delta_p in zip(deltas, deltas_p):
        for i in range(3):
            gamma_p = gammas[i] * (1 + etas[i] \
                * np.linalg.norm(delta_p - delta) / np.linalg.norm(delta))
            dHdxs[i] += gamma_p * np.exp(1j * Ka.dot(delta_p)) * 1j * delta_p[0]
            dHdys[i] += gamma_p * np.exp(1j * Ka.dot(delta_p)) * 1j * delta_p[1]
    dH0dx, dH3dx, dH4dx = dHdxs
    dH0dy, dH3dy, dH4dy = dHdys

    # Next-nearest neighbor matrix element
    dHndx = o
    dHndy = o
    for delta, delta_p in zip(deltans, deltans_p):
        gamma_p = c.gamman * (1 + etan \
            * np.linalg.norm(delta_p - delta) / np.linalg.norm(delta))
        dHndx += gamma_p * np.exp(1j * Ka.dot(delta_p)) * 1j * delta_p[0]
        dHndy += gamma_p * np.exp(1j * Ka.dot(delta_p)) * 1j * delta_p[1]

    gradH = [] # will contain dH/dx, dH/dy
    for dHn, dH0, dH3, dH4 in zip([dHndx, dHndy], [dH0dx, dH0dy], \
        [dH3dx, dH3dy], [dH4dx, dH4dy]):
        gradH.append(np.array([
            [dHn, dH3, -dH4.conj(), dH0.conj()],
            [dH3.conj(), dHn, dH0, -dH4],
            [-dH4, dH0.conj(), dHn, o],
            [dH0, -dH4.conj(), o, dHn]
        ]))
    return gradH
