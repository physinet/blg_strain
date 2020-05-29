import numpy as np

from .hamiltonian import H_dkx, H_dky
from .utils.const import q, hbar, muB

def berry_mu(E, Psi, xi=1):
    '''
    Calculates the Berry curvature and magnetic moment given the energy
    eigenvalues and eigenvectors for N(=4) bands.

    Params:
    - E: N(=4) x Nkx x Nky array of energy eigenvalues
    - Psi: N(=4) x N(=4) x Nkx x Nky array of energy eigenvectors.
      The first dimension indexes the band, and the second dimension indexes
      components of the eigenvectors.
    - xi: valley index (+1 for K, -1 for K')

    Returns:
    - Omega: n(=4) x Nkx x Nky array of Berry curvature (units m^2)
    - Mu: n(=4) x Nkx x Nky array of magnetic moment (units Bohr magneton)
    '''
    hdkx, hdky = H_dkx(xi), H_dky()  # 4x4 matrices

    Omega = np.zeros_like(E)  # 4 x Nkx x Nky
    Mu = np.zeros_like(E)

    for n, (e_n, psi_n) in enumerate(zip(E, Psi)):
        for m, (e_m, psi_m) in enumerate(zip(E, Psi)):
            if n == m: # sum runs over n != m
                continue

            # Calculate matrix products using einstein notation
            # These are inner products <n|H'|m> (H' derivative of Hamiltonian)
            # i and l index the components of the eigenvectors for bands n and m
            # the H' matrix is indexed with il to contract these indices
            # j and k index over the kx, ky points and are the dimensions left
            prod1 = np.einsum('ijk,il,ljk->jk', psi_n.conj(), hdkx, psi_m,
                                optimize=True)
            prod2 = np.einsum('ijk,il,ljk->jk', psi_m.conj(), hdky, psi_n,
                                optimize=True)

            Omega[n] += np.imag(prod1 * prod2 / (e_n - e_m) ** 2)
            Mu[n] += np.imag(prod1 * prod2 / (e_n - e_m))

    Omega = -2 * Omega
    Mu = -q / hbar * Mu / muB  # -> Bohr magnetons
                # [C / (eV * s)] * [(eV * m)^2 / eV] * [1 / A*m^2]
                # result should be dimensionless

    return Omega, Mu
