import numpy as np

from blg_strain.utils.const import mu0

def _parse_direction(x, y, direction):
    '''
    From a "direction" string ('+x', '-x', '+y', '-y'), returns the sign along
    with variables "s" and "t". If the direction contains x, s = x and t = y.
    If the direction contains y, s = y and t = x.
    '''
    assert direction in ('+x', '-x', '+y', '-y')
    if 'x' in direction:
        s = -y
        t = x
    else:
        s = x
        t = y
    if '+' in direction:
        sign = 1
    else:
        sign = -1
    return sign, s, t


def B_finite_wire(x, y, z, h, I, direction='+x'):
    '''
    Returns the z component of magnetic field Bz(x,y,z) for an infinitesimally
    wide, length-h wire segment with current I flowing in the given direction
    ('+x', '-x', '+y', '-y') centered at the origin.
    '''
    sign, s, t = _parse_direction(x, y, direction)

    prefactor = mu0 * I / (4 * np.pi) * s / (s ** 2 + z ** 2)
    term1 = (t - h / 2) / np.sqrt(s ** 2 + (t - h / 2) ** 2 + z ** 2)
    term2 = -(t + h / 2) / np.sqrt(s ** 2 + (t + h / 2) ** 2 + z ** 2)
    return sign * prefactor * (term1 + term2)


def B_wire_width(x, y, z, w, I, direction='+x'):
    '''
    Returns the z component of magnetic field Bz(x, y, z) for an infinite wire
    of width w with current I in the given direction ('+x', '-x', '+y', '-y')
    centered at the origin.

    If distances specified in meters and current in Ampere, returns magnetic
    field in Tesla
    '''
    sign, s, t = _parse_direction(x, y, direction)

    num = (s - w / 2) ** 2 + z ** 2
    denom = (s + w / 2) ** 2 + z ** 2
    return sign * mu0 * I / (4 * np.pi * w) * np.log(num / denom)


def B_mag_rect(x, y, z, w, h, M):
    '''
    Returns the z component of magnetic field Bz(x,y,z) for a 2D magnetized
    rectangle centered at the origin with width w and height h, with a 2D
    magnetization M.

    This calculation uses the fact that a uniform 2D magnetization can be
    replaced by a current of the same magnitude flowing at the boundary.

    If distance specified in meters and magnetization in Ampere, returns
    magnetic field in Tesla.
    '''
    Bright = B_finite_wire(x - w / 2, y, z, h, M, direction='+y')
    Btop = B_finite_wire(x, y - h / 2, z, w, M, direction='-x')
    Bleft = B_finite_wire(x + w / 2, y, z, h, M, direction='-y')
    Bbottom = B_finite_wire(x, y + h / 2, z, w, M, direction='+x')
    return Bright + Btop + Bleft + Bbottom
