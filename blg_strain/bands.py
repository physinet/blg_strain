import os
import numpy as np
from scipy.optimize import root

from .hamiltonian import H_4x4
from .berry import berry_mu
from .microscopic import feq_func
from .macroscopic import n_valley_layer, D_field, ME_coef
from .utils.utils import make_grid, get_splines, densify

from .utils.const import K, a0
from .utils.saver import Saver


def get_bands(Kxa, Kya, sl, eigh=True, ham='4x4', **params):
    '''
    Calculate energy eigenvalues and eigenvectors for a rectangular window of
    k-space.

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


class BandStructure(Saver):
    '''
    Contains calculated band structure around the K Dirac point for given
    strained lattice and choice of interlayer asymmetry.
    '''
    def __init__(self, sl=Saver(), window=0.1, Delta=0):
        '''
        sl: an instance of the `lattice.StrainedLattice` class
        window: region of K-space to sample (in units of 1/a0). A value of 0.1
            corresponds to 0.7 nm^-1.
        Delta: interlayer asymmetry (eV)
        '''
        self.sl = sl
        self.window = window
        self.Delta = Delta


    def calculate(self, Nkx=200, Nky=200, valley='K'):
        '''
        Calculate the band structure for one valley using the specified number
        of points Nkx, Nky for the Hamiltonian. Will calculate splines that can
        be used to interpolate to a greater density of samples.

        - Nkx, Nky: number of points used in grid to calculate band structure
        - Valley: either 'K' or 'Kp' - which valley to use for the calculation
        '''
        assert valley in ['K', 'Kp']
        self.valley = valley
        K = getattr(self.sl, valley)

        self.Nkx, self.Nky = Nkx, Nky
        kxalims = K[0] - self.window/2, K[0] + self.window/2
        kyalims = K[1] - self.window/2, K[1] + self.window/2
        self.kxa, self.kya, self.Kxa, self.Kya  = make_grid(kxalims, kyalims,
                                                            Nkx, Nky)
        self.E, self.Psi = get_bands(self.Kxa, self.Kya, self.sl,
                                Delta=self.Delta, ham='4x4', eigh=True)


        # Shift the zero energy point
        self.shift_zero_energy()

        self.Omega, self.Mu = berry_mu(self.Kxa, self.Kya, self.sl,
                                self.E, self.Psi)

        self._get_splines()


    def _densify(self, Nkx_new=2000, Nky_new=2000):
        '''
        Interpolate to dense grid

        Params:
        Nkx_new, Nky_new -  passed to densify - the final density of the grid
        '''
        self.kxa, self.kya, self.E, Pr, Pi, self.Omega, self.Mu = \
            densify(self.kxa, self.kya, self.splE, self.splPr, self.splPi,
                self.splO, self.splM, Nkx_new=Nkx_new, Nky_new=Nky_new)
        # combine real and imaginary parts - cast to lower precision to save mem
        self.Psi = np.array(Pr + 1j * Pi, dtype='complex64')
        # Meshgrid using ij indexing for compatibility with RectBivariateSpline
        self.Kxa, self.Kya = np.meshgrid(self.kxa, self.kya, indexing='ij')


    def _get_splines(self):
        '''
        Calculates the spline interpolations for various quantities. Run after
        reloading the object to recalculate the splines.
        '''
        self.splE, self.splPr, self.splPi, self.splO, self.splM = \
            get_splines(self.kxa, self.kya, self.E, self.Psi.real,
                self.Psi.imag, self.Omega, self.Mu)


    def shift_zero_energy(self):
        '''
        Calculates the zero energy point such that the net carrier density at
        EF = 0 is zero at zero temperature. Shifts the bands by an amount
        `self.E0` corresponding to this calculated zero energy point.
        '''
        # initial guess from average of conduction and valence band extrema
        E0 = np.mean([self.E[1].max(), self.E[2].min()])

        def n(EF):
            feq_K = feq_func(self.E, EF, T=0)
            # Factor of 2 for valley
            n1 = 2*n_valley_layer(self.kxa, self.kya, feq_K, self.Psi, layer=1)
            n2 = 2*n_valley_layer(self.kxa, self.kya, feq_K, self.Psi, layer=2)

            return (n1 + n2) / 1e16  # more natural units for solver

        result = root(n, E0)
        self.E0 = result.x
        self.E -= self.E0


    def save(self):
        '''
        Saves the object
        Delta in the filename reported in meV

        This will be saved in a subdirectory named after the StrainedLattice
        class.
        '''
        path = os.path.splitext(self.sl.filename)[0]
        filename = 'BandStructure_Nkx{:d}_Nky{:d}_Delta{:.3f}.h5'.format(
            self.Nkx, self.Nky, self.Delta * 1e3
        )
        self.filename = os.path.join(path, filename)

        skip = ['E', 'Omega', 'Mu', 'Psi']
        super().save(self.filename, skip=skip)


class FilledBands(Saver):
    '''
    Class to contain information derived from a band structure given a specified
    Fermi level EF and temperature T.
    '''
    def __init__(self, bs=Saver(), EF=0, T=0):
        '''
        Parameters:
        - bs: an instance of the `BandStructure` class
        - EF: Fermi level relative to the center of the band gap.
        - T: Temperature (K)
        '''
        self.bs = bs
        self.EF = EF
        self.T = T


    def calculate(self, Nkx=2000, Nky=2000):
        '''
        Calculate carrier density, displacement field, and magnetoelectric
        coefficient from the given band structure, Fermi level, and temperature.

        Nkx, NKy - num points to use in the grid to calculate ME coef
        '''

        bs = self.bs
        feq_K = feq_func(bs.E, self.EF, self.T)

        # Carrier density (m^-2) (contributions from each layer)
        # factor of 2 for valleys
        self.n1 = 2 * n_valley_layer(bs.kxa, bs.kya, feq_K, bs.Psi, layer=1)
        self.n2 = 2 * n_valley_layer(bs.kxa, bs.kya, feq_K, bs.Psi, layer=2)
        self.n = self.n1 + self.n2

        # Displacement field (V/m)
        self.D = D_field(bs.Delta, 2 * self.n1, 2 * self.n2)

        # Make dense grid
        self.Nkx, self.Nky = Nkx, Nky
        self.kxa, self.kya, = densify(bs.kxa, bs.kya, Nkx_new=Nkx, Nky_new=Nky)

        # ME coefficient with factor of 2 for valley
        # Skips calculating y component if strain is along major crystal axis
        if hasattr(self.bs.sl, 'theta'):
            dy = self.bs.sl.theta not in (0, np.pi/2)
        else:
            dy = True
        # dy = True  # Always calculate y component
        self.alpha = 2 * ME_coef(self.kxa, self.kya, bs.splE, bs.splO, bs.splM,
            self.EF, self.T, dy=dy)


    def save(self):
        '''
        Saves the object.
        EF in the filename reported in meV

        This will be saved in a subdirectory named after the BandStructure
        class.
        '''
        path = os.path.splitext(self.bs.filename)[0]
        filename = 'FilledBands_Nkx{:d}_Nky{:d}_EF{:.3f}_T{:.1f}.h5'.format(
            self.Nkx, self.Nky, self.EF*1e3, self.T
        )
        self.filename = os.path.join(path, filename)

        super().save(self.filename, compression=None)
