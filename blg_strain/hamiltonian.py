import numpy as np
from .utils.const import nu, eta0, eta3, eta4, etan, hbar, deltas, deltans
from .utils import const as c  # for accessing parameters we can turn on/off
from .utils.lattice import strain_tensor


def H_4by4(Kxa, Kya, Delta=0, eps=0, theta=0):
    '''
    Calculates the 4x4 low-energy Hamiltonian for uniaxially strained BLG.

    Parameters:
    - Kxa, Kya: Nkx x Nky array of wave vectors
    - Delta: interlayer asymmetry (eV)
    - eps: uniaxial strain
    - theta: angle of uniaxial strain to zigzag axis

    Returns:
    - H: Hamiltonian array shape 4 x 4 x Nkx x Nky
    '''
    # K vector
    Ka = np.stack([Kxa, Kya], axis=-1)

    o = np.zeros_like(Kxa, dtype='complex128') # Array to give  shape to const
    I = np.eye(2) # 2x2 identity matrix

    # strain tensor
    strain = strain_tensor(eps, theta)

    # Transform bonds under strain
    deltas_p = [(I+strain).dot(delta) for delta in deltas]
    deltans_p = [(I+strain).dot(deltan) for deltan in deltans]

    # Nearest-neighbor matrix elements
    gammas = [c.gamma0, c.gamma3, c.gamma4]
    etas = [eta0, eta3, eta4]
    Hs = np.array([o, o, o])
    for delta, delta_p in zip(deltas, deltas_p):
        for i in range(3):
            gamma_p = gammas[i] * (1 + etas[i] * np.linalg.norm(delta_p - delta) / np.linalg.norm(delta))
            Hs[i] += gamma_p * np.exp(1j * Ka.dot(delta_p))
    H0, H3, H4 = Hs

    # Next-nearest neighbor matrix element
    Hn = o
    for delta, delta_p in zip(deltans, deltans_p):
        gamma_p = c.gamman * (1 + etan * np.linalg.norm(delta_p - delta) / np.linalg.norm(delta))
        Hn += gamma_p * np.exp(1j * Ka.dot(delta_p))

    H = np.array([
        [-Delta / 2 + Hn, H3, -H4.conj(), H0.conj()],
        [H3.conj(), Delta/2 + Hn, H0, -H4],
        [-H4, H0.conj(), Delta/2 + Hn + c.DeltaAB, c.gamma1 + o],
        [H0, -H4.conj(), c.gamma1 + o, -Delta/2 + Hn + c.DeltaAB]
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
