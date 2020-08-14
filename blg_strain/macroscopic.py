import numpy as np
from scipy.integrate import simps
from scipy.interpolate import RectBivariateSpline

from .microscopic import check_f_boundaries, feq_func
from .utils.const import q, hbar, hbar_J, muB, mu0, eps0, d, a0

def n_valley_layer(kxa, kya, feq, Psi, layer=1):
    '''
    Calculate the contribution to the carrier density on layer 1/2 using the
    band structure for a single valley. This is an integral over the Brillouin
    zone of the occupation function weighted by the wavefunction amplitudes on
    each layer.

    Layer 1 corresponds to the first and last components of the (4-component)
    eigenvectors, while layer 2 corresponds to the middle two components.

    Parameters:
    - kxa, kya: Nkx, Nky arrays of kxa, kya points
    - feq: N(=4) x Nkx x Nky array of occupation for a given valley
    - Psi: N(=4) x N(=4) x Nkx x Nky array of eigenstates for a given valley
    - layer: layer number (1 or 2)
    '''
    assert feq.shape[0] == 4, 'Need 4x4 Hamiltonian to compute density by layer'
    assert layer in [1, 2]

    # The occupation should be zero at the edges of the window defined by kx,
    # ky. So the integration over the entire Brillouin zone should be captured
    # by integrating over this range of k's. Let's check that f is (nearly) zero
    # at the boundaries (this will print a message if not):
    check_f_boundaries(feq)

    if layer == 1:
        weight = abs(Psi[:, 0, :, :]) ** 2 + abs(Psi[:, 3, :, :]) ** 2
    else:
        weight = abs(Psi[:, 1, :, :]) ** 2 + abs(Psi[:, 2, :, :]) ** 2

    integrand = (2 * 2 / (2 * np.pi * a0) ** 2) * feq * weight
    # The integrand is an N(=4) x Nkx x Nky array
    # the inner simps integrates over ky (the last dimension of integrand)
    # the result of the inner integration is N(=4) x Nkx
    # the outer simps then integrates over kx to give a length N(=4) array
    integral = simps(simps(integrand, kya, axis=-1), kxa, axis=-1)

    # We now sum over the bands
    return integral.sum()


def n_layer(kxa, kya, feq1, feq2, Psi1, Psi2, layer=1):
    '''
    Calculate the contribution to the carrier density on layer 1/2 considering
    both valleys.

    Parameters:
    - kxa, kya: Nkx, Nky arrays of kxa, kya points
    - feq1, feq2: N(=4) x Nkx x Nky arrays of occupation for valley K and K'
    - Psi1, Psi2: N(=4) x N(=4) x Nkx x Nky arrays of eigenstates for K and K'
    - layer: layer number (1 or 2)
    '''
    assert layer in [1, 2]

    n = n_valley_layer(kxa, kya, feq1, Psi1, layer=layer) \
      + n_valley_layer(kxa, kya, feq2, Psi2, layer=layer)

    return n


def n_valley(kxa, kya, feq):
    '''
    Integrates the Fermi-Dirac distribution to calculate the total carrier
    density (in m^-2). This is the contribution from only one of the two valleys
    with energy eigenvalues given in E.

    Parameters:
    - kxa, kya: Nkx, Nky arrays of kxa, kya points
    - feq: N(=4) x Nkx x Nky array of occupation for valley K or K'
    - EF: Fermi energy (eV)
    - T: temperature (K)
    '''

    check_f_boundaries(feq)  # check if f is nearly zero at boundaries of region

    integrand = 2 * 2 / (2 * np.pi * a0) ** 2 * feq

    # integrate and sum over bands
    return simps(simps(integrand, kya, axis=-1), kxa, axis=-1).sum()


def ntot_func(kxa, kya, feq1, feq2):
    '''
    Integrates the Fermi-Dirac distribution to calculate the total carrier
    density (in m^-2). This is the sum of contributions from both valleys.

    Parameters:
    - kxa, kya: Nkx, Nky arrays of kxa, kya points
    - feq1, feq2: N(=4) x Nkx x Nky arrays of occupation for valley K and K'
    - EF: Fermi energy (eV)
    - T: temperature (K)
    '''
    return n_valley(kxa, kya, feq1) + n_valley(kxa, kya, feq2)


def D_field(Delta, nt, nb):
    '''
    Returns the electric displacement field D/epsilon0 across bilayer graphene
    corresponding to an interlayer asymmetry Delta and carrier density on each
    layer nt and nb. We divide by the vacuum permittivity epsilon0 and return
    the displacement field in units of electric field (V/m)

    Parameters:
    - Delta: interlayer asymmetry (eV)
    - nt (nb): carrier density on top (bottom) layer (m^-2)

    Returns:
    - D: electric displacement field (V/m)
    '''
    D = Delta / d - q / eps0 * (nt - nb)  # note Delta in eV,
                                          # so we leave out e in denominator
    return D  # V/m


def _ME_coef_integral(kxa, kya, splE, splO, splM, EF=0, T=0):
    '''
    Integrates over k space as part of the calculation for magnetoelectric
    coefficient for one valley and one band. This quantity is dimensionless
    and has x and y components. To calculate magnetization, compute the dot
    product with an electric field vector, multiply by a relaxation time, and
    divide by the vacuum permeability.

    Parameters:
    - kxa, kya: Nkx, Nky arrays of kxa, kya points
    - (splE, splO, splM) : splines for (energy / berry curvature / magnetic
        moment) for the given band
    - EF: Fermi energy (eV)
    - T: Temperature (K)

    Returns:
    - a length-2 array of x/y components of the dimensionless ME coefficient
    '''
    E = splE(kxa, kya)
    feq = feq_func(E, EF=EF, T=T)
    if (np.abs(feq).max() < 1e-4):  # Band unoccupied!
        return 0

    O = splO(kxa, kya)
    Mu = splM(kxa, kya)

    # non-equilibrium occupation ("divided by" dot product with E field)
    # we exclude equilibrium term that integrates to zero
    # note prefactor hbar is in J * s
    # factors of a0 to take care of integration and gradient w.r.t. k*a

    f = a0 * q * mu0  / (hbar_J) * np.array(np.gradient(feq, kxa, kya,
                                                        axis=(-2, -1)))

    integrand = 1 / (2 * np.pi * a0) ** 2 * f * (Mu + q * O / hbar * (EF - E))

    integral = simps(simps(integrand, kya, axis=-1), kxa, axis=-1)
    return integral


def _ME_coef_integral_by_parts(kxa, kya, splE, splO, splM, EF=0, T=0, dy=True):
    '''
    Integrates over k space as part of the calculation for magnetoelectric
    coefficient for one valley and one band. This quantity is dimensionless
    and has x and y components. To calculate magnetization, compute the dot
    product with an electric field vector, multiply by a relaxation time, and
    divide by the vacuum permeability.

    This is the ''integration by parts'' version in which the derivatives
    are moved to the energy, magnetic moment, and berry curvature terms.

    Parameters:
    - kxa, kya: Nkx, Nky arrays of kxa, kya points
    - (splE, splO, splM) : splines for (energy / berry curvature / magnetic
        moment) for the given band
    - EF: Fermi energy (eV)
    - T: Temperature (K)
    - dy: if False, assumes the y component integrates to zero (may speed up)

    Returns:
    - a length-2 array of x/y components of the dimensionless ME coefficient
    '''
    E = splE(kxa, kya)
    feq = feq_func(E, EF=EF, T=T)
    if (np.abs(feq).max() < 1e-4):  # Band unoccupied!
        return 0

    O = splO(kxa, kya)
    # Mu = splM(kxa, kya)

    E_dx = splE(kxa, kya, dx=1)
    O_dx = splO(kxa, kya, dx=1)
    Mu_dx = splM(kxa, kya, dx=1)

    # note prefactor hbar is in J * s
    # factors of a0 to take care of integration and gradient w.r.t. k*a
    prefactor = - a0 * q * mu0 / (hbar_J) / (2 * np.pi * a0) ** 2 * feq
    integrand = prefactor * (Mu_dx  \
                     + q / hbar * O_dx * (EF - E) \
                     - q / hbar * O * E_dx
                     )

    integralx = simps(simps(integrand, kya, axis=-1), kxa, axis=-1)

    if dy:
        E_dy = splE(kxa, kya, dy=1)
        O_dy = splO(kxa, kya, dy=1)
        Mu_dy = splM(kxa, kya, dy=1)

        integrand = prefactor * (Mu_dy  \
                         + q / hbar * O_dy * (EF - E) \
                         - q / hbar * O * E_dy
                         )
        integraly = simps(simps(integrand, kya, axis=-1), kxa, axis=-1)
    else:
        integraly = np.zeros_like(integralx)

    return np.array([integralx, integraly])


def ME_coef(kxa, kya, splE, splO, splM, EF=0, T=0, byparts=True, dy=True):
    '''
    Calculates the integral for ME coefficient for each of four bands and sums
    over bands. To calculate magnetization, compute the dot product with an
    electric field vector, multiply by a relaxation time, and divide by the
    vacuum permeability.

    Parameters:
    - kxa, kya: Nkx, Nky arrays of kxa, kya points
    - (splE, splO, splM) : N(=4) array of splines for (energy / berry curvature
        / magnetic moment) in each band
    - EF: Fermi energy (eV)
    - T: Temperature (K)
    - byparts: if True, will use the "integration by parts" version of the
        integral (function `_M_integral_by_parts`). If False, will use the
        function `_M_integral`
    - dy: if False, assumes the y component integrates to zero (speeds up calc)


    Returns:
    - a length-2 array of x/y components of the dimensionless ME coefficient
    '''
    N = len(splE)  # number of bands
    alpha = np.zeros((N, 2))  # (number of bands) x (x, y)

    if byparts:
        integral = _ME_coef_integral_by_parts
    else:
        integral = _ME_coef_integral

    for i in range(N):
        alpha[i] = integral(kxa, kya, splE[i], splO[i], splM[i], EF=EF, T=T,
            dy=dy)

    return alpha.sum(axis=0)  # sum over bands
