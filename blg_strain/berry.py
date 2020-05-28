import numpy as np
from numba import jit

from .hamiltonian import H_dkx, H_dky
from .utils.const import q, hbar, muB

@jit(parallel=True)
def berry_mu(E, Psi, xi=1):
    '''
    Calculates the Berry curvature and magnetic moment given the energy
    eigenvalues and eigenvectors for N(=4) bands.

    Params:
    - E: length N(=4) array of energy eigenvalues
    - Psi: N(=4) x N(=4) array of energy eigenvectors.
      The first dimension indexes the band, and the second dimension indexes
      components of the eigenvectors.
    - xi: valley index (+1 for K, -1 for K')

    Returns:
    - Omega: N(=4) array of Berry curvature (units m^2)
    - Mu: N(=4) array of magnetic moment (units Bohr magneton)
    '''
    hdkx, hdky = H_dkx(xi), H_dky()  # 4x4 matrices

    Omega = np.zeros_like(E) # length 4 array
    Mu = np.zeros_like(E)

    for n in range(E.shape[0]):
        for m in range(E.shape[0]):
            e_n = E[n]
            e_m = E[m]
            psi_n = np.ascontiguousarray(Psi[n])  # for numba jit
            psi_m = np.ascontiguousarray(Psi[m])

            if e_n == e_m: # sum runs over n != m
                continue

            num = psi_n.conjugate().dot(hdkx).dot(psi_m) * \
                  psi_m.conjugate().dot(hdky).dot(psi_n)

            denom = e_n - e_m

            Omega[n] += np.imag(num / (denom) ** 2)
            Mu[n] += np.imag(num / denom)

    Omega = -2 * Omega
    Mu = -q / hbar * Mu / muB  # -> Bohr magnetons
                # [C / (eV * s)] * [(eV * m)^2 / eV] * [1 / A*m^2]
                # result should be dimensionless

    return Omega, Mu
