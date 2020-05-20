import numpy as np
from scipy.integrate import simps
from .microscopic import feq_func, check_f_boundaries

def n_xi_func(kx, ky, E, Psi, EF, T=0, xi=1, layer=1):
    '''
    Calculate the contribution to the carrier density on layer 1/2 from valley
    K indexed by xi (K: 1, K': -1). This is an integral over the Brillouin zone
    of the occupation function weighted by the wavefunction amplitudes on each
    layer.

    In the K valley (xi=1), layer 1 corresponds to the first and last components
    of the (4-component) eigenvectors, while layer 2 corresponds to the middle
    two components. For the K' valley (xi=-1), this is reversed.
    '''
    assert layer in [1,2]

    feq = feq_func(E, EF, T)
    # This should be zero at the edges of the window defined by kx, ky. So the
    # integration over the entire Brillouin zone should be captured by
    # integrating over this range of k's. Let's check that f is (nearly) zero
    # at the boundaries (this will print a message if not):
    check_f_boundaries(feq)

    if (layer == 1 and xi == 1) or (layer == 2 and xi == -1):
        weight = abs(Psi[:, 0, :, :])**2 + abs(Psi[:, 3, :, :])**2
    elif (layer == 2 and xi == 1) or (layer == 1 and xi == -1):
        weight = abs(Psi[:, 1, :, :])**2 + abs(Psi[:, 2, :, :])**2

    integrand = (2 * 2 / (2 * np.pi) ** 2) * feq * weight
    # The integrand is an N(=4) x Nky x Nkx array
    # the inner simps integrates over kx (the last dimension of integrand)
    # the result of the inner integration is N(=4) x Nky
    # the outer simps then integrates over ky to give a length N(=4) array
    integral = simps(simps(integrand, kx, axis=-1), ky, axis=-1)

    # We now sum over the bands
    return integral.sum()

def n_func(kx, ky, E, Psi, EF, T=0, layer=1):
    '''
    Sums `n_xi_func` for each valley to get the total carrier density on the
    specified layer.
    '''
    n = 0
    for xi in [1, -1]:
        n += n_xi_func(kx, ky, E, Psi, EF, T=T, xi=xi, layer=layer)
    return n

def ntot_func(kx, ky, E, Psi, EF, T=0):
    '''
    Sums `n_func` for each layer to get total carrier density.
    '''
    n = 0
    for layer in [1, 2]:
        n += n_func(kx, ky, E, Psi, EF, T=T, layer=layer)
    return n
