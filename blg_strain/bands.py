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
    - kx, ky: Nkx, Nky arrays of kx, ky points
    - Kx, Ky: Nky x Nkx meshgrid of kx, ky points
    - E: N(=4) x Nky x Nkx array of energy eigenvalues
    - Psi: N(=4) x N(=4) x Nky x Nkx array of eigenvectors
    - Omega: N(=4) x Nky x Nkx array of berry curvature
    - Mu: N(=4) x Nky x Nkx array of magnetic moment
    '''
    kx = np.linspace(kxlims[0], kxlims[1], Nkx)
    ky = np.linspace(kylims[0], kylims[1], Nky)

    Kx, Ky = np.meshgrid(kx, ky)

    E, Psi, Omega, Mu = _get_bands(Kx, Ky, xi=xi, Delta=Delta, delta=delta,
                        theta=theta)

    return kx, ky, Kx, Ky, E, Psi, Omega, Mu

def _get_bands(Kx, Ky, xi=1, **params):
    '''
    Calculate energy eigenvalues, eigenvectors, berry curvature, and magnetic
    moment for a rectangular window of k-space.

    Parameters:
    - Kx, Ky: Nky x Nkx meshgrid of kx, ky points
    - xi: valley index (+1 for K, -1 for K')
    - params: passed to `hamiltonian.H_func`

    Returns:
    - E: N(=4) x Nky x Nkx array of energy eigenvalues
    - Psi: N(=4) x N(=4) x Nky x Nkx array of eigenvectors
    - Omega: N(=4) x Nky x Nkx array of berry curvature
    - Mu: N(=4) x Nky x Nkx array of magnetic moment
    '''
    #
    # E = np.zeros((4, len(ky), len(kx)))
    # Psi = np.zeros((4, 4, len(ky), len(kx)), dtype='complex')
    # Omega = np.zeros((4, len(ky), len(kx)))
    # Mu = np.zeros((4, len(ky), len(kx)))

    H = Hfunc(Kx, Ky, xi=xi, **params)

    H = H.swapaxes(0, 2).swapaxes(1,3) # put the 4x4 in the last 2 dims for eigh
    E, Psi = np.linalg.eigh(H)  # using eigh for Hermitian
                                # eigenvalues are real and sorted (low to high)
    # Shapes - E: Nky x Nkx x 4, Psi: Nky x Nkx x 4 x 4
    E = E.swapaxes(0,1).swapaxes(0,2) # put the kx,ky points in last 2 dims
    Psi = Psi.swapaxes(0, 2).swapaxes(1,3) # put the kx,ky points in last 2 dims
    # now E[:, 0, 0] is a length-4 array of eigenvalues
    # and Psi[:, :, 0, 0] is a 4x4 array of eigenvectors (in the columns)

    # Address sign ambiguity
    # Force the first component of each eigenvector to be positive
    # Create a multiplier: 1 if the first component is positive
    #                     -1 if the first component is negative
    multiplier = 2 * (Psi[0, :, :, :].real > 0) - 1  # shape 4 x Nky x Nkx
    Psi *= multiplier  # flips sign of eigenvectors where 1st component is -ive
                       # NOTE: broadcasting rules dictate that the multiplier is
                       # applied to all components of the first dimension, thus
                       # flipping the sign of the entire vector

    if (Psi[0, :, :, :].real < 0).any():  # double check that all positive
        raise Exception('Fixing sign ambiguity failed! There are eigenvectors \
                with a negative first component!')

    # Finally, transpose first 2 dimensions to put the eigenvectors in the rows
    Psi = Psi.transpose((1,0,2,3))

    # Now E[n, 0, 0] and Psi[n, :, 0, 0] give the energy and eigenstates

    # Omega, Mu = berry_mu(E, Psi, xi=xi)
    Omega, Mu = None, None #TODO

    return E, Psi, Omega, Mu
