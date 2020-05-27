import numpy as np

from .const import eta0, eta3, eta4, gamma3, gamma4, nu

def w(delta, idx=3, xi=1, theta=0):
    '''
    Calculate gauge fields w3/w4 from strain.

    idx: 3 or 4
    '''
    assert idx in [3,4]
    assert xi in [1, -1]

    if idx == 3:
        gamma = gamma3
        eta = eta3
    elif idx == 4:
        gamma = gamma4
        eta = eta4
    return 3 / 4 * np.exp(-1j * 2 * xi * theta) * (1 + nu) \
                 * delta * (eta - eta0) * gamma
