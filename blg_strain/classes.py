import numpy as np

from .bands import get_bands
from .berry import berry_mu
from .macroscopic import _M_bands, n_valley_layer, disp_field
from .microscopic import feq_func
from .utils.utils import get_splines, densify
from .utils.const import gamma4, dab

class Valley:
    '''
    Class to contain results of calculations for one valley
    '''
    def __init__(self, xi=1, **kwargs):
        '''
        xi - valley index (1 or -1)
        kwargs passed to get_bands
        '''
        assert xi in [1, -1], 'Valley index must be either 1 or -1!'
        self.xi = xi

        self.kwargs = kwargs

    def _calculate(self, Nkx_new=2000, Nky_new=2000):
        '''
        Nkx_new, Nky_new passed to densify - the final density of the grid

        Get bands, etc. and interpolate to dense grid
        '''
        # Initial calculations with the number of points given in kwargs
        self._kx, self._ky, self._Kx, self._Ky, self._E, self._Psi = \
            get_bands(xi=self.xi, **self.kwargs)

        H_gradient=None
        if 'ham' in self.kwargs:
            if self.kwargs['ham'] not in ['2x2', '4x4']:
                H_gradient = np.gradient(self.kwargs['ham'], self._kx, \
                    self._ky, axis=(-2,-1))

        self._Omega, self._Mu = berry_mu(self._Kx, self._Ky, self._E, self._Psi,
            xi=self.xi, H_gradient=H_gradient)

        # Calculate spline interpolations and dense grid
        self.splE, splPr, splPi, self.splO, self.splM = \
            get_splines(self._kx, self._ky, self._E, self._Psi.real,
                self._Psi.imag, self._Omega, self._Mu)
        self.kx, self.ky, self.E, Pr, Pi, self.Omega, self.Mu = \
            densify(self._kx, self._ky, self.splE, splPr, splPi, self.splO,
                self.splM, Nkx_new=Nkx_new, Nky_new=Nky_new)
        self.Psi = Pr + 1j * Pi
        self.Kx, self.Ky = np.meshgrid(self.kx, self.ky, indexing='ij')


class BandStructure:
    '''
    Class to contain band structure. Quantities for each valley are stored in
    the variables `BandStructure.K` and `BandStructure.Kp`.
    '''
    _default_valley='K'  # can access the calculated quantities for this valley
                         # (e.g. Omega) by simply writing `BandStructure.Omega`

    def __init__(self, **kwargs):
        '''
        kwargs passed to get_bands
        '''
        for key in kwargs:
            setattr(self, key, kwargs[key])

        self.K = Valley(xi=1, **kwargs)
        self.Kp = Valley(xi=-1, **kwargs)

        # Parameters that we may alter
        self.gamma4 = gamma4
        self.dab = dab

    def __getattr__(self, attr):
        '''
        Attempts to get attributes from `self.K` or `self.Kp`
        '''
        return getattr(getattr(self, self._default_valley), attr)


    def calculate(self, Nkx_new=2000, Nky_new=2000):
        '''
        Nkx_new, Nky_new passed to densify - the final density of the grid

        Perform calculations for both valleys and interpolate to dense grid
        '''
        self.K._calculate(Nkx_new=Nkx_new, Nky_new=Nky_new)
        self.Kp._calculate(Nkx_new=Nkx_new, Nky_new=Nky_new)


    def get_nD(self):
        '''
        Calculates the total carrier density and displacement field (parameters
        that can be tuned experimentally). This requires calculating the carrier
        density for each valley/layer combination. These quantities are stored,
        for example, in `BandStructure.K.n1` for the density on layer 1 in
        valley K. The carrier density and displacement field are stored in
        `BandStructure.n` (units m^-2) and `BandStructure.D` (units V / m).
        '''
        if not hasattr(self.K, 'feq'):
            raise Exception('Calculate feq using `get_feq` first!')

        for v in (self.K, self.Kp):
            for i in [1,2]:
                setattr(v, 'n%i' %i, n_valley_layer(v.kx, v.ky, v.feq, v.Psi,
                    layer=i))

        self.n = self.K.n1 + self.K.n2 + self.Kp.n1 + self.Kp.n2
        self.D = disp_field(self.Delta, self.K.n1 + self.Kp.n1,
            self.K.n2 + self.Kp.n2)


    def get_feq(self, EF, T=0):
        '''
        Calculates the equilibrium Fermi occupation function for both valleys
        and stores results under (e.g.) `BandStructure.K.feq`

        EF: Fermi energy (eV)
        T: temperature (K)
        '''
        for valley in (self.K, self.Kp):
            setattr(valley, 'EF', EF)
            setattr(valley, 'T', T)
            setattr(valley, 'feq', feq_func(valley.E, EF, T))


    def get_M(self, Efield = [0,0]):
        '''
        Computes the dot product of `M_over_E` with an electric field provided
        as a length-2 array of x and y components (in V / m). Returns
        magnetization in Bohr magneton / um^2.
        '''
        if not hasattr(self, 'M_over_E'):
            raise Exception('Calculate M_over_E using `get_M_over_E first!`')

        return self.M_over_E.dot(Efield)


    def get_M_over_E(self, tau=1e-12):
        '''
        Calculates the net magnetization summing contributions from both
        valleys. This quantity, when dotted with an electric field (in V/m)
        gives the orbital magnetization in Bohr magneton / um^2

        tau: relaxation time (seconds). Default: 1 picosecond
        '''
        if not hasattr(self.K, 'feq'):
            raise Exception('Calculate feq using `get_feq` first!')

        self.M_over_E = 0
        for v in (self.K, self.Kp):
            self.M_over_E += _M_bands(v.kx, v.ky, v.feq, v.splE, v.splO, v.splM,
                tau=tau, EF=v.EF, byparts=True)

        return self.M_over_E


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


    def set_default_valley(self, valley):
        '''
        valley: either 'K' or 'Kp' to indicate which valley's variables (e.g.
        Omega) should be accessed with `BandStructure.Omega` (for example)
        '''

        assert valley in ['K', 'Kp'], 'Valley must be either \'K\' or \'Kp\''
        self._default_valley = valley
