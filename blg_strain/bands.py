import numpy as np

from .hamiltonian import H_4x4
from .berry import berry_mu
from .utils.utils import make_grid, get_splines, densify

from .utils.const import K



def get_bands(sl, kxalims=[-1.2 * K, 1.2 * K], kyalims=[-1.2 * K, 1.2 * K], Nkx=200,
                Nky=200, Delta=0, ham='4x4', eigh=False):
    '''
    Calculate energy eigenvalues and eigenvectors for a rectangular window of
    k-space.

    Parameters:
    - sl: instance of the StrainedLattice class
    - kxalims, kyalims: length-2 arrays of min and max wave vectors (units 1/a0)
        By default, covers the entire Brillouin zone (with a little extra)
    - Nkx, Nky: number of k points in each dimension
    - Delta: interlayer asymmetry (eV)
    - ham: str or array - Select choice of Hamiltonian with a string
        (choice is only '4x4' for now) or pass precompuuted array consistent
        with the arrays made with `make_grid(kxlims, kylims, Nkx, Nky)`.
        Shape is N x N (x Nkx x Nky), where the last two dimensions are included
        if the Hamiltonian varies with kx and ky.
    - eigh: if True, use np.linalg.eigh; if False use np.linalg.eig

    Returns:
    - kxa, kya: Nkx, Nky arrays of kx, ky points
    - Kxa, Kya: Nkx x Nky meshgrid of kx, ky points (note: we use `ij` indexing
        compatible with scipy spline interpolation methods)
    - E: N(=4) x Nkx x Nky array of energy eigenvalues
    - Psi: N(=4) x N(=4) x Nkx x Nky array of eigenvectors
    '''
    kxa, kya, Kxa, Kya = make_grid(kxalims, kyalims, Nkx, Nky)

    E, Psi = _get_bands(Kxa, Kya, sl, Delta=Delta, ham=ham,
                        eigh=eigh)

    return kxa, kya, Kxa, Kya, E, Psi


def _get_bands(Kxa, Kya, sl, eigh=True, ham='4x4', **params):
    '''
    Calculate energy eigenvalues and eigenvectors for a rectangular window of
    k-space. This function is wrapped by `blg_strain.bands.get_bands`.

    Parameters:
    - Kxa, Kya: Nkx x Nky meshgrid of kx, ky points (using 'ij' indexing)
    - eigh: if True, use np.linalg.eigh; if False use np.linalg.eig
    - hamiltonian: str or array - Select choice of Hamiltonian with a string
        (only '4x4' for now) or pass array of precomputed Hamiltonian
        with shape N x N x Nkx x Nky.
    - params: passed to `hamiltonian.H_4x4`

    Returns:
    - E: N(=4) x Nkx x Nky array of energy eigenvalues
    - Psi: N(=4) x N(=4) x Nkx x Nky array of eigenvectors
    '''
    assert type(ham) in (str, np.ndarray), 'ham param should be str or array'
    if type(ham) is str:
        assert ham in ('4x4',), 'ham parameter must be \'4x4\', or you must \
         pass an array'
        if ham == '4x4':
            H = H_4x4(Kxa, Kya, sl, **params)
        elif ham == '2x2':
            raise Exception('4x4 only for now')
    else:
        H = ham  # an array

    H = H.transpose(2,3,0,1) # put the 4x4 in the last 2 dims for eigh

    # check if Hermitian
    if not np.allclose(H, H.transpose(0,1,3,2).conj()):
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


class Valley:
    '''
    Contains calcualted band structure around one Dirac point for given
    strained lattice and choice of interlayer asymmetry.
    '''
    def __init__(self, sl, Delta=0, valley='K'):
        '''
        sl: an instance of the `lattice.StrainedLattice` class
        Delta: interlayer asymmetry (eV)
        valley: either 'K' or 'Kp' - choose valley to calculate band structure
            around. The locations of each valley are stored in sl.K and sl.Kp.
        '''
        self.sl = sl
        self.Delta = Delta
        self.valley = valley


    def _calculate(self, Nkx=200, Nky=200, Nkx_new=2000, Nky_new=2000):
        '''
        Get bands, etc. and interpolate to dense grid

        Params:
        Nkx_new, Nky_new -  passed to densify - the final density of the grid
        '''
        # Initial calculations with the number of points given in kwargs
        K = getattr(self.sl, self.valley)
        kxalims = K[0] - 0.1, K[0] + 0.1
        kyalims = K[1] - 0.1, K[1] + 0.1
        self._kxa, self._kya, self._Kxa, self._Kya, self._E, self._Psi = \
            get_bands(self.sl, kxalims=kxalims, kyalims=kyalims, Nkx=Nkx,
                Nky=Nky, Delta=self.Delta)

        self._Omega, self._Mu = berry_mu(self._Kxa, self._Kya, self.sl,
                                self._E, self._Psi)

        # Calculate spline interpolations and dense grid
        self.splE, splPr, splPi, self.splO, self.splM = \
            get_splines(self._kxa, self._kya, self._E, self._Psi.real,
                self._Psi.imag, self._Omega, self._Mu)
        self.kxa, self.kya, self.E, Pr, Pi, self.Omega, self.Mu = \
            densify(self._kxa, self._kya, self.splE, splPr, splPi, self.splO,
                self.splM, Nkx_new=Nkx_new, Nky_new=Nky_new)
        self.Psi = Pr + 1j * Pi  # separate splines for real and imaginary parts
        # Meshgrid using ij indexing for compatibility with RectBivariateSpline
        self.Kxa, self.Kya = np.meshgrid(self.kxa, self.kya, indexing='ij')


class BandStructure:
    '''
    Contains calculated band structure around the Dirac points (simply labeled
    K and K') for given strained lattice and choice of interlayer asymmetry.
    '''
    def __init__(self, sl, Delta=0):
        '''
        sl: an instance of the `lattice.StrainedLattice` class
        Delta: interlayer asymmetry (eV)
        '''
        self.sl = sl
        self.Delta = Delta

        self.K = Valley(sl, Delta, 'K')
        self.Kp = Valley(sl, Delta, 'Kp')


    def calculate(self, Nkx_new=2000, Nky_new=2000):
        '''
        Nkx_new, Nky_new passed to densify - the final density of the grid

        Perform calculations for both valleys and interpolate to dense grid
        '''
        self.K._calculate(Nkx_new=Nkx_new, Nky_new=Nky_new)
        self.Kp._calculate(Nkx_new=Nkx_new, Nky_new=Nky_new)


    @classmethod
    def load(cls, filename):
        '''
        Returns a BandStructure class object with parameters loaded from the
        .npz file at location `filename`.
        '''
        obj = cls()  # inizialize class object

        data = np.load(filename, allow_pickle=True)

        # Set saved variables as attributes to the class object
        for attr in data.files:
            setattr(obj, attr, data[attr].item())

        return obj


    def save(self, filename):
        '''
        Saves data to a compressed .npz file

        filename: full path of destination file
        '''
        if filename[-4:] != '.npz':
            filename += '.npz'

        np.savez_compressed(filename, **self.__dict__)
