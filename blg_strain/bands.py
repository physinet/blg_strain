import numpy as np

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
    - Kx, Ky: Nky x Nkx meshgrid of kx, ky points
    - E: N(=4) x Nky x Nkx array of energy eigenvalues
    - Psi: N(=4) x N(=4) x Nky x Nkx array of eigenvectors
    - Omega: N(=4) x Nky x Nkx array of berry curvature
    - Mu: N(=4) x Nky x Nkx array of magnetic moment
    '''
    kx = np.linspace(kxlims[0], kxlims[1], Nkx)
    ky = np.linspace(kylims[0], kylims[1], Nky)

    Kx, Ky = np.meshgrid(kx, ky)

    E, Psi, Omega, Mu = _get_bands(kx, ky, xi=xi, Delta=Delta, delta=delta,
                        theta=theta)

    return Kx, Ky, E, Psi, Omega, Mu

def _get_bands(kx, ky, xi=1, **params):
    '''
    Calculate energy eigenvalues, eigenvectors, berry curvature, and magnetic
    moment for a rectangular window of k-space.

    Parameters:
    - Kx, Ky: Nky x Nkx meshgrid of kx, ky points
    - params: passed to
    - xi: valley index (+1 for K, -1 for K')
    - params: passed to `hamiltonian.H_func`

    Returns:
    - E: N(=4) x Nky x Nkx array of energy eigenvalues
    - Psi: N(=4) x N(=4) x Nky x Nkx array of eigenvectors
    - Omega: N(=4) x Nky x Nkx array of berry curvature
    - Mu: N(=4) x Nky x Nkx array of magnetic moment
    '''

    E = np.zeros((4, len(ky), len(kx)))
    Psi = np.zeros((4, 4, len(ky), len(kx)), dtype='complex')
    Omega = np.zeros((4, len(ky), len(kx)))
    Mu = np.zeros((4, len(ky), len(kx)))

    for i, x in enumerate(kx):
        for j, y in enumerate(ky):
            h = Hfunc(x, y, xi=xi, **params)

            eigs, vecs = np.linalg.eigh(h) # Use eigh for Hermitian
             # guaranteed sorted real eigenvalues
            vecs = vecs.T # eigenvectors are in the columns
                          # -> transpose to put in rows

            # Fix sign ambiguity
            for k in range(len(vecs)):
                if vecs[k][0].real < 0: # check sign of first component
                    vecs[k] *= -1 # guarantee first component is positive

            E[:, j, i] = eigs # Energy for the two bands
            Psi[:, :, j, i] = vecs # eigenstates

            Omega[:, j, i], Mu[:, j, i] = berry_mu(eigs, vecs, xi=xi)

    return E, Psi, Omega, Mu
