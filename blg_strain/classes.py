import numpy as np

from .bands import get_bands
from .berry import berry_mu
from .macroscopic import ntot_func, M_valley
from .microscopic import feq_func
from .utils.utils import get_splines, densify

class BandStructure:
    '''
    Class to contain results of calculations involving only the Hamiltonian
    '''
    def __init__(self, xi=1, **kwargs):
        '''
        xi - valley index (1 or -1)
        kwargs passed to get_bands
        '''
        assert xi in [1, -1], 'Valley index must be either 1 or -1!'
        self.xi = xi

        self.kwargs = kwargs

    def calculate(self, Nkx_new=2000, Nky_new=2000):
        '''
        Nkx_new, Nky_new passed to densify - the final density of the grid

        Get bands, etc. and interpolate to dense grid
        '''
        # Initial calculations with the number of points given in kwargs
        self._kx, self._ky, self._Kx, self._Ky, self._E, self._Psi = \
            get_bands(xi=self.xi, **self.kwargs)

        if 'ham' in self.kwargs:
            H_gradient = np.gradient(self.kwargs['ham'], self._kx, self._ky, \
                axis=(-2,-1))
        else:
            H_gradient=None

        self._Omega, self._Mu = berry_mu(self._Kx, self._Ky, self._E, self._Psi,
            xi=self.xi, H_gradient=H_gradient)

        # Calculate spline interpolations and dense grid
        self.splE, self.splO, self.splM = \
            get_splines(self._kx, self._ky, self._E, self._Omega, self._Mu)
        self.kx, self.ky, self.E, self.Omega, self.Mu = \
            densify(self._kx, self._ky, self.splE, self.splO, self.splM, \
                Nkx_new=Nkx_new, Nky_new=Nky_new)
        self.Kx, self.Ky = np.meshgrid(self.kx, self.ky, indexing='ij')


class BothValleys:
    '''
    Class to contain results of calculations involving variable Fermi energy
    '''
    def __init__(self, bs1, bs2, EF=0, T=0):
        '''
        bs1, bs2: BandStructure objects for K and K' valley
        '''
        self.bs1, self.bs2, self.EF, self.T = bs1, bs2, EF, T

    def get_f(self):
        self.feq1 = feq_func(self.bs1.E, self.EF, self.T)
        self.feq2 = feq_func(self.bs2.E, self.EF, self.T)

    def get_n(self):
        self.n = ntot_func(self.bs1.kx, self.bs1.ky, self.feq1, self.feq2)

    def get_M(self):
        self.M1 = M_valley(self.bs1.kx, self.bs1.ky, self.feq1, self.bs1.splE,
            self.bs1.splO, self.bs1.splM, Efield=[1,0], tau=1, EF=self.EF)
