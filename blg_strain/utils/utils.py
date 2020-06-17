import numpy as np
from scipy.interpolate import RectBivariateSpline

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


def get_splines(kx, ky, *args):
    '''
    For all args (defined as ... x Nkx x Nky arrays), returns arrays containing
    spline interpolations for each array in args.

    Returns:
    *splines - unpacked tuple of arrays corresponding to each arg. The shape of
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
            splines[i][idx] = RectBivariateSpline(kx, ky, arg[idx])

    if len(splines) == 1:
        return splines[0]
    else:
        return splines


def make_grid(kxlims=[-0.35e9, .35e9], kylims=[-0.35e9, .35e9], Nkx=200,
                Nky=200):
    kx = np.linspace(kxlims[0], kxlims[1], Nkx)
    ky = np.linspace(kylims[0], kylims[1], Nky)

    Kx, Ky = np.meshgrid(kx, ky, indexing='ij')

    return kx, ky, Kx, Ky
