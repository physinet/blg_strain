import numpy as np
from scipy.interpolate import RectBivariateSpline
import datetime
from .saver import Saver
from skimage.measure import find_contours

class Spline(RectBivariateSpline, Saver):
    '''
    Custom Spline class to enable saving and loading
    '''
    def __init__(self):
        pass

    def calculate(self, x, y, z):
        super().__init__(x,y,z)

    def save(self, filename):
        '''
        Extract spline parameters information from the `tck` tuple to enable
        saving as individual arrays
        '''
        self.tx, self.ty, self.c = self.tck
        super().save(filename)

    @classmethod
    def load(cls, filename):
        '''
        Load `Spline` object and reconstruct `tck` parameter
        '''
        obj = super().load(filename)
        obj.tck = obj.tx, obj.ty, obj.c
        return obj


def densify(kx, ky, *args, Nkx_new=1000, Nky_new=1000):
    '''
    For all args (which are arrays of spline interpolations), evaluates the
    splines over a dense grid of kx, ky points spanning the same ranges as the
    inputs but with Nkx_new, Nky_new points in the grid

    Returns:
    kxdense, kydense - length Nkx_new or Nky_new array over the same range of
        values as kx and ky
    *newargs - unpacked tuple of args evaluated over the grid, with new shape
        ... x Nkx x Nky
    '''
    kxdense = np.linspace(kx.min(), kx.max(), Nkx_new)
    kydense = np.linspace(ky.min(), ky.max(), Nky_new)
    # Kxdense, Kydense = np.meshgrid(kxdense, kydense, indexing='ij')

    returns = [None] * len(args)

    for i, arg in enumerate(args):
        returns[i] = np.empty((*arg.shape, Nkx_new, Nky_new))

        # iterate over all dimensions
        for idx in np.ndindex(arg.shape):
            returns[i][idx] = arg[idx](kxdense, kydense)  # evaluate on new grid

    return (kxdense, kydense, *returns)


def get_contours(kxa, kya, E, EF):
    '''
    Get contour E=EF for an energy band

    Parameters:
    - kxa, kya: Nkx, Nky arrays of kxa, kya points
    - E: Nkx x Nky array of energy (eV)
    - EF: Fermi level (eV)

    Returns:
    contours: list of (n,2)-ndarrays,
        Each contour is an ndarray of shape ``(n, 2)``,
        consisting of n ``(row, column)`` coordinates along the contour.
    '''
    contours = find_contours(E, EF)

    # Transform to kxa, kya coordinates
    scalex = kxa.ptp() / (len(kxa) - 1)
    scaley = kya.ptp() / (len(kya) - 1)
    for c in contours:
        c[:, 0] = kxa.min() + c[:, 0] * scalex
        c[:, 1] = kya.min() + c[:, 1] * scaley

    return contours


def get_splines(kx, ky, *args):
    '''
    For all args (defined as ... x Nkx x Nky arrays), returns arrays containing
    spline interpolations for each array in args.

    Returns:
    * splines - unpacked tuple of arrays corresponding to each arg. The shape of
        each array is equal to arg.shape[:-2]
    '''
    splines = [None] * len(args)

    for i, arg in enumerate(args):
        assert arg.shape[-2] == len(kx)
        assert arg.shape[-1] == len(ky)

        # Empty array with shape of all but last two dimensions of arg
        splines[i] = np.empty(arg.shape[:-2], dtype='object')

        # iterate over all but last two dimensions
        for idx in np.ndindex(arg.shape[:-2]):
            s = Spline()
            s.calculate(kx, ky, arg[idx])
            splines[i][idx] = s

    if len(splines) == 1:
        return splines[0]
    else:
        return splines


def make_grid(xlims=[-0.35e9, .35e9], ylims=[-0.35e9, .35e9], Nx=200,
                Ny=200):
    kx = np.linspace(xlims[0], xlims[1], Nx)
    ky = np.linspace(ylims[0], ylims[1], Ny)

    Kx, Ky = np.meshgrid(kx, ky, indexing='ij')

    return kx, ky, Kx, Ky


def print_time(d={}):
    '''
    Prints the current timestamp along with a dictionary of parameters values
    and/or messages.

    For example, `print_time({'Status': 'Running...', 'Delta': 2})` might print:
        `2020-07-02 18:02:24    Status: Running...      Delta: 2`
    '''
    now = datetime.datetime.now()
    s = now.strftime('%Y-%m-%d %H:%M:%S')
    for k, v in d.items():
        s += '\t'
        s += '{}: {}'.format(k, v)

    print(s)
