import numpy as np
from numba import jit

from .berry import berry_mu
from .hamiltonian import Hfunc

def get_bands(kxlims=[-0.35e9, .35e9], kylims=[-0.35e9, .35e9], Nkx=200,
                Nky=200, xi=1, Delta=0, delta=0, theta=0):
    '''
    Calculate energy eigenvalues, eigenvectors, berry curvature, and magnetic
    moment for a rectangular window of k-space.

    Parameters:
    - kxlims, kylims: length-2 arrays of min and max wave vectors (nm^-1)
    - Nkx, Nky: number of k points in each dimension
    - xi: valley index (+1 for K, -1 for K')
    - Delta: interlayer asymmetry (eV)
    - delta: uniaxial strain
    - theta: angle of uniaxial strain to zigzag axis

    Returns:
    - kx, ky: Nkx, Nky arrays of kx, ky points
    - Kx, Ky: Nkx x Nky meshgrid of kx, ky points (note: we use `ij` indexing
        compatible with scipy spline interpolation methods)
    - E: N(=4) x Nkx x Nky array of energy eigenvalues
    - Psi: N(=4) x N(=4) x Nkx x Nky array of eigenvectors
    - Omega: N(=4) x Nkx x Nky array of berry curvature
    - Mu: N(=4) x Nkx x Nky array of magnetic moment
    '''
    kx = np.linspace(kxlims[0], kxlims[1], Nkx)
    ky = np.linspace(kylims[0], kylims[1], Nky)

    Kx, Ky = np.meshgrid(kx, ky, indexing='ij')

    E, Psi, Omega, Mu = _get_bands(kx, ky, xi=xi, Delta=Delta, delta=delta,
                            theta=theta)

    return kx, ky, Kx, Ky, E, Psi, Omega, Mu


@jit(parallel=True)
def _get_bands(kx, ky, xi=1, Delta=0, delta=0, theta=0):
    '''
    Calculate energy eigenvalues, eigenvectors, berry curvature, and magnetic
    moment for a rectangular window of k-space.

    Parameters:
    - Kx, Ky: Nkx x Nky meshgrid of kx, ky points (using 'ij' indexing)
    - xi: valley index (+1 for K, -1 for K')
    - Delta: interlayer asymmetry (eV)
    - delta: uniaxial strain
    - theta: angle of uniaxial strain to zigzag axis

    Returns:
    - E: N(=4) x Nkx x Nky array of energy eigenvalues
    - Psi: N(=4) x N(=4) x Nkx x Nky array of eigenvectors
    - Omega: N(=4) x Nky x Nkx array of berry curvature
    - Mu: N(=4) x Nky x Nkx array of magnetic moment
    '''

    E = np.zeros((4, len(kx), len(ky)))
    Psi = np.zeros((4, 4, len(kx), len(ky)), dtype=np.complex128)
    Omega = np.zeros((4, len(kx), len(ky)))
    Mu = np.zeros((4, len(kx), len(ky)))

    for i in range(len(kx)):
        for j in range(len(ky)):
            # Make x and y 2-dimensional for compatibility with Hfunc
            x = kx[i:i+1].reshape(1,1)  # the i:i+1 is necessary for jit to work
            y = ky[j:j+1].reshape(1,1)

            h = Hfunc(x, y, xi=xi, Delta=Delta, delta=delta,
                theta=theta)[:, :, 0, 0]
                  # Hfunc returns shape (4, 4, 1, 1)

            eigs, vecs = np.linalg.eigh(h) # Use eigh for Hermitian
             # guaranteed sorted real eigenvalues
            vecs = vecs.T # eigenvectors are in the columns
                          # -> transpose to put in rows

            # Fix sign ambiguity
            for k in range(len(vecs)):
                if vecs[k][0].real < 0: # check sign of first component
                    vecs[k] *= -1 # guarantee first component is positive

            E[:, i, j] = eigs # Energy for the two bands
            Psi[:, :, i, j] = vecs # eigenstates

            Omega[:, i, j], Mu[:, i, j] = berry_mu(eigs, vecs, xi=xi)

    return E, Psi, Omega, Mu
