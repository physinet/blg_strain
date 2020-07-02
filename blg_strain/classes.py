import numpy as np

from .bands import get_bands
from .berry import berry_mu
from .macroscopic import ntot_func, M_valley
from .microscopic import feq_func
from .utils.utils import get_splines, densify

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
        self.splE, self.splO, self.splM = \
            get_splines(self._kx, self._ky, self._E, self._Omega, self._Mu)
        self.kx, self.ky, self.E, self.Omega, self.Mu = \
            densify(self._kx, self._ky, self.splE, self.splO, self.splM, \
                Nkx_new=Nkx_new, Nky_new=Nky_new)
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
        self.kwargs = kwargs
        self.K = Valley(xi=1, **kwargs)
        self.Kp = Valley(xi=-1, **kwargs)


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


    def set_default_valley(self, valley):
        '''
        valley: either 'K' or 'Kp' to indicate which valley's variables (e.g.
        Omega) should be accessed with `BandStructure.Omega` (for example)
        '''

        assert valley in ['K', 'Kp'], 'Valley must be either \'K\' or \'Kp\''
        self._default_valley = valley
