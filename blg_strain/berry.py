import numpy as np

from .hamiltonian import H_dkx, H_dky, H2_dkx, H2_dky
from .utils.const import q, hbar, muB
from .utils.utils import get_splines

def berry_mu(Kx, Ky, E, Psi, xi=1, einsum=True):
    '''
    Calculates the Berry curvature and magnetic moment given the energy
    eigenvalues and eigenvectors for N(=4) bands.

    Params:
    - Kx, Ky: Nkx x Nky array of wave vectors (nm^-1)
    - E: N(=4) x Nkx x Nky array of energy eigenvalues
    - Psi: N(=4) x N(=4) x Nkx x Nky array of energy eigenvectors.
      The first dimension indexes the band, and the second dimension indexes
      components of the eigenvectors.
    - xi: valley index (+1 for K, -1 for K')
    - einsum: if True, use `np.einsum` to multiply (faster than `np.tensordot`)

    Returns:
    - Omega: n(=4) x Nkx x Nky array of Berry curvature (units m^2)
    - Mu: n(=4) x Nkx x Nky array of magnetic moment (units Bohr magneton)
    '''
    if E.shape[0] == 2:
        hdkx, hdky = H2_dkx(Kx, Ky, xi), H2_dky(Kx, Ky, xi)
    else:
        hdkx, hdky = H_dkx(xi), H_dky(xi)  # 4x4 matrices

    Omega = np.zeros_like(Psi, dtype='float')  # 4 x 4 x Nkx x Nky; first dim
    Mu = np.zeros_like(Psi, dtype='float')     # summed over bands m != n

    for n, (e_n, psi_n) in enumerate(zip(E, Psi)):
        for m, (e_m, psi_m) in enumerate(zip(E, Psi)):
            if n == m: # sum runs over m != n
                continue

            if einsum:
                # Calculate matrix products using einstein notation
                # These are inner products <n|H'|m> (H' derivative of Hamiltonian)
                # i and l index the components of the eigenvectors for bands n and m
                # the H' matrix is indexed with il to contract these indices
                # j and k index over the kx, ky points and are the dimensions left
                if hdkx.ndim == 2:  # 4x4
                    prod1 = np.einsum('ijk,il,ljk->jk', psi_n.conj(), hdkx, psi_m,
                                optimize=True)
                    prod2 = np.einsum('ijk,il,ljk->jk', psi_m.conj(), hdky, psi_n,
                                optimize=True)
                else:  # 2x2 - also depends on kx and ky so shape is 2 x 2 x Nkx x Nky
                    prod1 = np.einsum('ijk,iljk,ljk->jk', psi_n.conj(), hdkx, psi_m,
                                optimize=True)
                    prod2 = np.einsum('ijk,iljk,ljk->jk', psi_m.conj(), hdky, psi_n,
                                optimize=True)
            else:
                # Equivalent method (slower!)
                # We want C = A * H * B, or c_jk = A_ijk * H_il * B_ljk
                # First sum over l using tensordot
                # Dot product over LAST axis of H and FIRST axis of B
                D1 = np.tensordot(hdkx, psi_m, axes=([-1], [0]))
                D2 = np.tensordot(hdky, psi_n, axes=([-1], [0]))
                # Now we have C = A * D, or c_jk = A_ijk * D_ijk
                # We want to sum over i only, so multiply then sum over axis 0
                prod1 = np.sum(np.multiply(psi_n.conj(), D1), axis=0)
                prod2 = np.sum(np.multiply(psi_m.conj(), D2), axis=0)

            Omega[m, n] = np.imag(np.multiply(prod1, prod2) / (e_n - e_m) ** 2)
            Mu[m, n] = np.imag(np.multiply(prod1, prod2) / (e_n - e_m))

    Omega = Omega.sum(axis=0)  # perform the sum over bands
    Mu = Mu.sum(axis=0)

    Omega = -2 * Omega
    Mu = -q / hbar * Mu / muB  # -> Bohr magnetons
                # [C / (eV * s)] * [(eV * m)^2 / eV] * [1 / A*m^2]
                # result should be dimensionless

    return Omega, Mu


def berry_connection(kx, ky, splPr, splPi):
    '''
    Returns the Berry connection A = -i<n|∇|n> from the eigenstates of the
    diagonalized Hamiltonian. This may not work with eigenstates calculated
    using np.linalg.eigh because they are not smoothly differentiable.
    Eigenstates calculated using the slower np.linalg.eig appear to work.

    Params:
    - kx, ky: Nkx, Nky arrays of kx, ky points
    - splPr, splPi: N(=4) x N(=4) arrays of splines for energy eigenvectors.
        splPr contains the real part, splPi contains the imaginary part.

    Returns:
    - Ax, Ay: N(=4) x Nkx x Nky arrays for x,y components of Berry connection
        (z component is zero)
    '''
    Ax = np.empty((splPr.shape[0], len(kx), len(ky)), dtype='complex')
    Ay = np.empty_like(Ax)

    for n in range(splPr.shape[0]): # Index over bands
        psi = np.empty_like(Ax)
        psi_dkx = np.empty_like(Ax)
        psi_dky = np.empty_like(Ax)

        for m in range(splPr.shape[1]): # Index over components of eigenvector
            psi[m] = splPr[n,m](kx, ky) + 1j * splPi[n,m](kx, ky)

            psi_dkx[m] = splPr[n,m](kx, ky, dx=1) + 1j \
                        * splPi[n,m](kx, ky, dx=1)
            psi_dky[m] = splPr[n,m](kx, ky, dy=1) + 1j \
                        * splPi[n,m](kx, ky, dy=1)

        Ax[n] = 1j * np.einsum('ijk,ijk->jk', psi.conjugate(), psi_dkx, \
                    optimize=True)
        Ay[n] = 1j * np.einsum('ijk,ijk->jk', psi.conjugate(), psi_dky, \
                    optimize=True)

    return Ax, Ay


def berry_from_connection(kx, ky, Ax, Ay):
    '''
    Calculates Berry curvature ∇ x A given the components of the Berry
        connection A.

    Params:
    - kx, ky: Nkx, Nky arrays of kx, ky points
    - Ax, Ay: N(=4) x Nkx x Nky arrays for x,y components of Berry connection

    Returns:
    - Oz: N(=4) x Nkx x Nky array of Berry curvature (z component; others zero)
    '''
    sxr, sxi, syr, syi = get_splines(kx, ky, Ax.real, Ax.imag, Ay.real, Ay.imag)
    Oz = np.empty_like(Ax)
    for n in range(Ax.shape[0]):
        Oz[n] = syr[n](kx, ky, dx=1) - sxr[n](kx, ky, dy=1)
        Oz[n] += 1j * (syi[n](kx, ky, dx=1) - sxi[n](kx, ky, dy=1))

    return Oz
