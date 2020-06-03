import numpy as np
from scipy.integrate import simps
from scipy.interpolate import RectBivariateSpline

from .microscopic import check_f_boundaries
from .utils.const import q, hbar, muB

def n_valley_layer(kx, ky, feq, Psi, EF, T=0, xi=1, layer=1):
    '''
    Calculate the contribution to the carrier density on layer 1/2 from valley
    K or K' indexed by xi (K: 1, K': -1). This is an integral over the Brillouin
    zone of the occupation function weighted by the wavefunction amplitudes on
    each layer.

    In the K valley (xi=1), layer 1 corresponds to the first and last components
    of the (4-component) eigenvectors, while layer 2 corresponds to the middle
    two components. For the K' valley (xi=-1), this is reversed.
    '''
    assert xi in [1, -1]
    assert layer in [1, 2]

    # The occupation should be zero at the edges of the window defined by kx,
    # ky. So the integration over the entire Brillouin zone should be captured
    # by integrating over this range of k's. Let's check that f is (nearly) zero
    # at the boundaries (this will print a message if not):
    check_f_boundaries(feq)

    if (layer == 1 and xi == 1) or (layer == 2 and xi == -1):
        weight = abs(Psi[:, 0, :, :]) ** 2 + abs(Psi[:, 3, :, :]) ** 2
    elif (layer == 2 and xi == 1) or (layer == 1 and xi == -1):
        weight = abs(Psi[:, 1, :, :]) ** 2 + abs(Psi[:, 2, :, :]) ** 2

    integrand = (2 * 2 / (2 * np.pi) ** 2) * feq * weight
    # The integrand is an N(=4) x Nkx x Nky array
    # the inner simps integrates over ky (the last dimension of integrand)
    # the result of the inner integration is N(=4) x Nkx
    # the outer simps then integrates over kx to give a length N(=4) array
    integral = simps(simps(integrand, ky, axis=-1), kx, axis=-1)

    # We now sum over the bands
    return integral.sum()


def n_layer(kx, ky, feq1, feq2, Psi1, Psi2, EF, T=0, layer=1):
    '''
    Calculate the contribution to the carrier density on layer 1/2 considering
    both valleys.

    Parameters:
    - kx, ky: Nkx, Nky arrays of kx, ky points
    - feq1, feq2: N(=4) x Nkx x Nky arrays of occupation for valley K and K'
    - Psi1, Psi2: N(=4) x N(=4) x Nkx x Nky arrays of eigenstates for K and K'
    - EF: Fermi energy (eV)
    - T: temperature (K)
    - layer: layer number (1 or 2)
    '''
    assert layer in [1, 2]

    n = n_valley_layer(kx, ky, feq1, Psi1, EF, T=T, layer=layer) \
      + n_valley_layer(kx, ky, feq2, Psi2, EF, T=T, layer=layer)

    return n


def n_valley(kx, ky, feq, EF, T=0):
    '''
    Integrates the Fermi-Dirac distribution to calculate the total carrier
    density (in m^-2). This is the contribution from only one of the two valleys
    with energy eigenvalues given in E.

    Parameters:
    - kx, ky: Nkx, Nky arrays of kx, ky points
    - feq: N(=4) x Nkx x Nky array of occupation for valley K or K'
    - EF: Fermi energy (eV)
    - T: temperature (K)
    '''

    check_f_boundaries(feq)  # check if f is nearly zero at boundaries of region

    integrand = 2 * 2 / (2 * np.pi) ** 2 * feq

    # integrate and sum over bands
    return simps(simps(integrand, ky, axis=-1), kx, axis=-1).sum()


def ntot_func(kx, ky, feq1, feq2, EF, T=0):
    '''
    Integrates the Fermi-Dirac distribution to calculate the total carrier
    density (in m^-2). This is the sum of contributions from both valleys.

    Parameters:
    - kx, ky: Nkx, Nky arrays of kx, ky points
    - feq1, feq2: N(=4) x Nkx x Nky arrays of occupation for valley K and K'
    - EF: Fermi energy (eV)
    - T: temperature (K)
    '''
    return n_valley(kx, ky, feq1, EF, T=T) + n_valley(kx, ky, feq2, EF, T=T)


def M_func_K(kx, ky, E, Omega, Mu, Efield=[0,0], tau=0, EF=0, T=0):
    '''
    Integrates over k space to get orbital magnetization for one valley.

    Parameters:
    - kx, ky: Nkx, Nky arrays of kx, ky points
    - E: N(=4) x Nkx x Nky array of energy eigenvalues
    - Omega: N(=4) x Nkx x Nky array of berry curvature
    - Mu: N(=4) x Nkx x Nky array of magnetic moment (in Bohr magnetons)
    - Efield: length-2 array of electric field x/y components (V/m)
    - tau: scattering time (s). In general an Nkx x Nky array.
    - EF: Fermi energy (eV)
    - T: temperature (K)
    '''
    Ex, Ey = np.array(Efield)

    feq = feq_func(E, EF, T)

    E_dkx, E_dky = np.gradient(E, kx, ky, axis=(-2,-1))
    Omega_dkx, Omega_dky = np.gradient(Omega, kx, ky, axis=(-2,-1))
    Mu_dkx, Mu_dky = np.gradient(Mu, kx, ky, axis=(-2,-1))

    integrandx = - q * tau * Ex / hbar / (2 * np.pi) ** 2 * feq * \
                 (Mu_dkx * muB + q * Omega_dkx / hbar * (EF-E) \
                               - q * Omega / hbar * E_dkx)
    integrandy = - q * tau * Ey / hbar / (2 * np.pi) ** 2 * feq * \
                 (Mu_dky * muB + q * Omega_dky / hbar * (EF-E) \
                               - q * Omega / hbar * E_dky)

    integral = simps(simps(integrandx + integrandy, ky, axis=-1), kx, axis=-1)

    return integral.sum(axis=0) # sum over bands


def D_valley(kx, ky, splE, splO, EF=0, T=0):
    '''
    Integrates over k space to get Berry curvature dipole for one valley.
    Integral is not summed over bands!

    Parameters:
    - kx, ky: Nkx, Nky arrays of kx, ky points
    - splE: N(=4) array of splines for energy eigenvalues for valley K or K'
    - splO: N(=4) array of splines for berry curvature in each band
    - EF: Fermi energy (eV)
    - T: temperature (K)
    '''
    N = splE.shape[0]  # 4
    D = np.empty(N)

    for n in range(N):
        f = feq_func(splE[n](kx, ky), EF, T)
        Omega_dkx = splO[n](kx, ky, dx=1)
        D[n] = simps(simps(Omega_dkx * f, ky), kx)
            # integral over y (axis -1) then x (axis -1 of the result of simps)

    return D  # not yet summed over bands


def D_func(kx, ky, splE1, splE2, splO1, splO2, EF=0, T=0):
    '''
    Integrates over k space to get Berry curvature dipole. This is the sum of
    contributions from both valleys. Integral is not summed over bands!

    Parameters:
    - kx, ky: Nkx, Nky arrays of kx, ky points
    - splE1, splE2: N(=4) arrays of splines for energy for valley K and K'
    - splO1, splO2: N(=4) array of splines for berry curvature in K and K'
    - EF: Fermi energy (eV)
    - T: temperature (K)
    '''
    return D_valley(kx, ky, splE1, splO1, EF, T=T) \
         + D_valley(kx, ky, splE2, splO2, EF, T=T)
