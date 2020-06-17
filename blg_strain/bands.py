import numpy as np

from .berry import berry_mu
from .hamiltonian import H_4by4, H_2by2
from .utils.utils import make_grid

def get_bands(kxlims=[-0.35e9, .35e9], kylims=[-0.35e9, .35e9], Nkx=200,
                Nky=200, xi=1, Delta=0, delta=0, theta=0, ham='4x4',
                eigh=False):
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
    - ham: str or array - Select choice of Hamiltonian with a string
        (choices are '4x4' or '2x2') or pass array of precomputed Hamiltonian.
        Shape is N x N (x Nkx x Nky), where the last two dimensions are included
        if the Hamiltonian varies with kx and ky.
    - eigh: if True, use np.linalg.eigh; if False use np.linalg.eig

    Returns:
    - kx, ky: Nkx, Nky arrays of kx, ky points
    - Kx, Ky: Nkx x Nky meshgrid of kx, ky points (note: we use `ij` indexing
        compatible with scipy spline interpolation methods)
    - E: N(=4) x Nkx x Nky array of energy eigenvalues
    - Psi: N(=4) x N(=4) x Nkx x Nky array of eigenvectors
    '''
    kx, ky, Kx, Ky = make_grid(kxlims, kylims, Nkx, Nky)

    E, Psi = _get_bands(Kx, Ky, xi=xi, Delta=Delta, delta=delta,
                        theta=theta, ham=ham, eigh=eigh)

    return kx, ky, Kx, Ky, E, Psi


def _get_bands(Kx, Ky, xi=1, eigh=True, ham='4x4', **params):
    '''
    Calculate energy eigenvalues, eigenvectors, berry curvature, and magnetic
    moment for a rectangular window of k-space.

    Parameters:
    - Kx, Ky: Nkx x Nky meshgrid of kx, ky points (using 'ij' indexing)
    - xi: valley index (+1 for K, -1 for K')
    - eigh: if True, use np.linalg.eigh; if False use np.linalg.ei
    - hamiltonian: str or array - Select choice of Hamiltonian with a string
        (choices are '4x4' or '2x2') or pass array of precomputed Hamiltonian
        with shape N x N x Nkx x Nky.
    - params: passed to `hamiltonian.H_4by4` or `hamiltonian.H_2by2`

    Returns:
    - E: N(=4) x Nkx x Nky array of energy eigenvalues
    - Psi: N(=4) x N(=4) x Nkx x Nky array of eigenvectors
    '''
    assert type(ham) in (str, np.ndarray), 'ham param should be str or array'
    if type(ham) is str:
        assert ham in ('4x4', '2x2'), 'ham parameter must be \'2x2\' or \
            \'4x4\', or you must pass an array'
        if ham == '4x4':
            H = H_4by4(Kx, Ky, xi=xi, **params)
        elif ham == '2x2':
            H = H_2by2(Kx, Ky, xi=xi, **params)
    else:
        H = ham  # an array

    H = H.transpose(2,3,0,1) # put the 4x4 in the last 2 dims for eigh

    # check if Hermitian
    if not (H == H.transpose(0,1,3,2).conjugate()).all():
        raise Exception('Hamiltonian is not Hermitian! Cannot use eigh.')

    if eigh:
        E, Psi = np.linalg.eigh(H)  # using eigh for Hermitian
                                # eigenvalues are real and sorted (low to high)
    else:
        E, Psi = np.linalg.eig(H)  # Option to use eig instead of eigh

    # Shapes - E: Nkx x Nky x 4, Psi: Nkx x Nky x 4 x 4
    E = E.transpose(2,0,1) # put the kx,ky points in last 2 dims
    Psi = Psi.transpose(2,3,0,1) # put the kx,ky points in last 2 dims
    # now E[:, 0, 0] is a length-4 array of eigenvalues
    # and Psi[:, :, 0, 0] is a 4x4 array of eigenvectors (in the columns)

    if eigh:
        Psi = fix_first_component_sign(Psi)

    else:
        E, Psi = sort_eigen(E, Psi)
        E = E.real




    # Finally, transpose first 2 dimensions to put the eigenvectors in the rows
    Psi = Psi.transpose((1,0,2,3))

    # Now E[n, 0, 0] and Psi[n, :, 0, 0] give the energy and eigenstates

    return E, Psi


def sort_eigen(eigs, vecs):
    '''
    Sort eigenvalues and the corresponding eigenvectors in order of increasing
    eigenvalues.

    Params:
    - eigs: N x ... array of eigenvalues
    - vecs: N x N x ... array of eigenvectors (in the columns)

    Returns:
    - eigs, vecs: sorted arrays in order of increasing eigenvalues
    '''
    indE = np.indices(eigs.shape)  # Get the indices such that
                                   # eigs[tuple(indE)] == eigs
    indV = np.indices(vecs.shape)

    indE[0] = eigs.argsort(axis=0)  # Sort the indices along the first axis
    indV[1] = indE[0]  # vectors sorted along *second* axis

    return eigs[tuple(indE)], vecs[tuple(indV)]


def fix_first_component_sign(Psi):
    '''
    Addresses sign ambiguity in eigenvectors by focing first component positive
    '''
    # Create a multiplier: 1 if the first component is positive
    #                     -1 if the first component is negative
    multiplier = 2 * (Psi[0].real > 0) - 1  # shape 4 x Nkx x Nky
    Psi *= multiplier  # flips sign of eigenvectors where 1st component is -ive
                       # NOTE: broadcasting rules dictate that the multiplier is
                       # applied to all components of the first dimension, thus
                       # flipping the sign of the entire vector

    if (Psi[0].real < 0).any():  # double check that all positive
        raise Exception('Fixing sign ambiguity failed! There are eigenvectors \
                with a negative first component!')

    return Psi
