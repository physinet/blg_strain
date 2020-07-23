import numpy as np
from numpy import sin, cos
import itertools
from scipy.spatial import Voronoi

from .const import nu, K

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
    I = np.eye(2)
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
