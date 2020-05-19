import numpy as np

from .hamiltonian import H_dkx, H_dky
from .utils.const import q, hbar, muB

def berry_mu(E, Psi, xi=1):
    '''
    Calculates the Berry curvature and magnetic moment given the energy
    eigenvalues and eigenvectors for N(=4) bands.

    Params:
    - E: N(=4) x Nky x Nkx array of energy eigenvalues
    - Psi: N(=4) x N(=4) x Nky x Nkx array of energy eigenvectors.
      The first dimension indexes the band, and the second dimension indexes
      components of the eigenvectors.
    - xi: valley index (+1 for K, -1 for K')

    Returns:
    - Omega: n(=4) x Nky x Nkx array of Berry curvature (units m^2)
    - Mu: n(=4) x Nky x Nkx array of magnetic moment (units Bohr magneton)
    '''
    hdkx, hdky = H_dkx(xi), H_dky()

    # Convert to matrices to make multiplication easier
    hdkx = np.matrix(hdkx)
    hdky = np.matrix(hdky)

    Omega = np.empty(E.shape[0]) # length 4 array
    Mu = np.empty(E.shape[0])

    for n, (e_n, psi_n) in enumerate(zip(E, Psi)):
        psi_n = np.matrix(psi_n).T # transpose to a column vector
        for e_m, psi_m in zip(E, Psi):
            if e_n == e_m: # sum runs over n != m
                continue

            psi_m = np.matrix(psi_m).T # transpose to a column vector

            # Using .H for Hermitian conjugate
            # .item() extracts the number from the 1x1 resulting matrix
            num = ((psi_n.H @ hdkx @ psi_m) * (psi_m.H @ hdky @ psi_n)).item()
            denom = e_n - e_m

            Omega[n] += np.imag(num / (denom) ** 2)
            Mu[n] += np.imag(num / denom)

    Omega = -2 * Omega
    Mu = -q / hbar * Mu / muB  # -> Bohr magnetons
                # [C / (eV * s)] * [(eV * m)^2 / eV] * [1 / A*m^2]
                # result should be dimensionless

    return Omega, Mu
