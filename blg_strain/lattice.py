import numpy as np
from numpy import sin, cos
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import itertools
from scipy.spatial import Voronoi
from scipy.optimize import minimize
from scipy.interpolate import RectBivariateSpline

from .utils.const import K, deltas, deltans, nu, eta0, eta3, eta4, etan, hbar, \
                            gamma0, gamma3, gamma4, gamman, DeltaAB
from .bands import get_bands

I = np.eye(2)

def strain_tensor(eps, theta):
    '''
    Returns the 4x4 strain tensor for uniaxial strain of magnitude eps
    applied at an angle theta measured from the +x axis.
    '''
    return eps * np.array([
        [cos(theta)**2 - nu * sin(theta)**2, (1+nu)*cos(theta)*sin(theta)],
        [(1+nu)*cos(theta)*sin(theta), sin(theta)**2 - nu * cos(theta)**2]
    ])


def brillouin_zone(strain):
    '''
    Returns the Brillouin zone distorted by a given strain tensor.
    Transforms the lattice basis vectors with strain tensor, then constructs
    the Brillouin zone in the usual way.
    '''
    # lattice basis vectors
    a1 = np.array([np.sqrt(3)/2, 3/2])
    a2 = np.array([-np.sqrt(3)/2, 3/2])

    # strained basis vectors
    a1p = (I + strain).dot(a1)
    a2p = (I + strain).dot(a2)

    # Reciprocal lattice vectors
    z = [0, 0, 1]
    b1p = 2 * np.pi * np.cross(a2p, z) / np.linalg.norm(np.cross(a1p, a2p))
    b2p = 2 * np.pi * np.cross(z, a1p) / np.linalg.norm(np.cross(a1p, a2p))

    return _get_bz_vertices([b1p, b2p])


def _get_bz_vertices(recip_vectors):
    '''
    Get the vertices of the Brillouin zone given list of reciprocal lattice
    vectors.
    See https://github.com/dean0x7d/pybinding/blob/master/pybinding/lattice.py
    '''
    points = [sum(n * v for n, v in zip(ns, recip_vectors))
                      for ns in itertools.product([-1, 0, 1], repeat=2)]
    vor = Voronoi([p[:2] for p in points])
    # See scipy's Voronoi documentation for details (-1 indicates infinity)
    finite_regions = [r for r in vor.regions if len(r) != 0 and -1 not in r]
    assert len(finite_regions) == 1
    return [vor.vertices[i] for i in finite_regions[0]]


def strained_K(strain, Kprime=False):
    '''
    Get the location of the K (or K') point of the strained Brillouin zone.
    Finds the vertex of the strained Brillouin zone that is closest to the
    location of the original K (or K') point.
    Parameters:
    - strain: an arbitrary 2x2 strain tensor
    - Kprime: if False (True), returns the position of the K (K') point
    '''
    bz = np.array(brillouin_zone(strain))  # list to array
    KK = np.array([K, 0])
    if Kprime:
        KK *= -1

    dist2 = ((bz - KK) ** 2).sum(axis=1)  # Dist between K point and bz vertices
    arg = np.argmin(dist2)
    return bz[arg]  # closest vertex to K (K')


class StrainedLattice:
    '''
    Class to contain one strain state. The most information stored here includes
    the strain tensor, the changes in bond lengths and hopping parameters due
    to strain, and the new location of the Dirac points under strain.
    '''
    def __init__(self, eps=0, theta=0):
        '''
        eps: uniaxial strain magnitude
        theta: uniaxial strain direction
        '''
        # Strain tensor
        self.strain = strain_tensor(eps, theta)

        # Find shifted K and K' points
        self._get_valleys()

        # Calculate hopping parameters with all parameters turned on
        self._calc_hopping()

        # Calculate brillouin zone vertices
        self.bz = brillouin_zone(self.strain)


    def _calc_hopping(self, turn_off=[]):
        '''
        Calculate changes in bond lengths and hopping parameters under strain.
        Use the turn_off argument to turn parameters off. This is useful for
        temporarily ignoring the effects of the smaller hopping parameters.
        List of parameters that can be turned off:
            gamma0, gamma3, gamma4, gamman, DeltaAB
        turn_off: list of parameter names to turn off. For example, ['gamma3'].
        '''

        self.deltas = []
        self.deltans = []
        self.gamma0s = []
        self.gamma3s = []
        self.gamma4s = []
        self.gammans = []
        self.DeltaAB = DeltaAB

        for delta in deltas:
            deltap = (I + self.strain).dot(delta)

            gamma0p = gamma0 * (1 + eta0 * np.linalg.norm(deltap - delta) \
                                              / np.linalg.norm(delta))
            gamma3p = gamma3 * (1 + eta3 * np.linalg.norm(deltap - delta) \
                                              / np.linalg.norm(delta))
            gamma4p = gamma4 * (1 + eta4 * np.linalg.norm(deltap - delta) \
                                              / np.linalg.norm(delta))

            self.deltas.append(deltap)
            self.gamma0s.append(gamma0p)
            self.gamma3s.append(gamma3p)
            self.gamma4s.append(gamma4p)

        for delta in deltans:
            deltap = (I + self.strain).dot(delta)

            gammanp = gamman * (1 + etan * np.linalg.norm(deltap - delta) \
                                              / np.linalg.norm(delta))

            self.deltans.append(deltap)
            self.gammans.append(gammanp)

        for param in turn_off:
            if param == 'DeltaAB':
                self.DeltaAB = 0
            else:
                exec('self.{0}s = [0 for g in self.{0}s]'.format(param))


    def _get_valleys(self):
        '''
        Calculate band structure with only gamma0, gamma1 turned on and locate
        the position of the Dirac points under strain. These do not coincide
        with the high-symmetry points of the Brillouin zone and must be
        found from the band structure extrema.
        '''

        # Calculate band structure with Delta = 0 and some hoppings turned off
        self._calc_hopping(turn_off=['gamma3', 'gamma4', 'gamman', 'DeltaAB'])

        kxa, kya, Kxa, Kya, E, Psi = get_bands(self, kxalims=[-1.2*K, 1.2*K],
            kyalims=[-1.2*K, 1.2*K], Nkx=200, Nky=200)

        # K and K' points of the strained Brillouin zone
        self.K_bz = strained_K(self.strain, Kprime=False)
        self.Kp_bz = strained_K(self.strain, Kprime=True)

        # Find minima
        spl = RectBivariateSpline(kxa, kya, E[2])  # spline function
        def f(x): # for compatibility with minimize
            return spl(*x)

        res = minimize(f, self.K_bz)
        self.K = res.x
        res = minimize(f, self.Kp_bz)
        self.Kp = res.x

    def plot_bz(self, ax):
        '''
        Draws the Brillouin zone and Dirac points on the given axis object ax.
        '''
        p = Polygon(self.bz, fill=False, color='k', lw=2)
        ax.add_patch(p)
        ax.plot(*self.K, 'or')
        ax.plot(*self.Kp, 'or')
