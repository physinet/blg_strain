import numpy as np
from scipy.integrate import simps
from scipy.interpolate import RectBivariateSpline

from .microscopic import check_f_boundaries
from .utils.const import q, hbar, muB, eps0, d

def n_valley_layer(kx, ky, feq, Psi, layer=1):
    '''
    Calculate the contribution to the carrier density on layer 1/2 using the
    band structure for a single valley. This is an integral over the Brillouin
    zone of the occupation function weighted by the wavefunction amplitudes on
    each layer.

    Layer 1 corresponds to the first and last components of the (4-component)
    eigenvectors, while layer 2 corresponds to the middle two components.

    Parameters:
    - kx, ky: Nkx, Nky arrays of kx, ky points
    - feq: N(=4) x Nkx x Nky array of occupation for a given valley
    - Psi: N(=4) x N(=4) x Nkx x Nky array of eigenstates for a given valley
    - layer: layer number (1 or 2)
    '''
    assert feq.shape[0] == 4, 'Layer density needs 4x4 eigenstates'
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

    integrand = (2 * 2 / (2 * np.pi) ** 2) * feq * weight
    # The integrand is an N(=4) x Nkx x Nky array
    # the inner simps integrates over ky (the last dimension of integrand)
    # the result of the inner integration is N(=4) x Nkx
    # the outer simps then integrates over kx to give a length N(=4) array
    integral = simps(simps(integrand, ky, axis=-1), kx, axis=-1)

    # We now sum over the bands
    return integral.sum()


def n_layer(kx, ky, feq1, feq2, Psi1, Psi2, layer=1):
    '''
    Calculate the contribution to the carrier density on layer 1/2 considering
    both valleys.

    Parameters:
    - kx, ky: Nkx, Nky arrays of kx, ky points
    - feq1, feq2: N(=4) x Nkx x Nky arrays of occupation for valley K and K'
    - Psi1, Psi2: N(=4) x N(=4) x Nkx x Nky arrays of eigenstates for K and K'
    - layer: layer number (1 or 2)
    '''
    assert layer in [1, 2]

    n = n_valley_layer(kx, ky, feq1, Psi1, layer=layer) \
      + n_valley_layer(kx, ky, feq2, Psi2, layer=layer)

    return n


def n_valley(kx, ky, feq):
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


def ntot_func(kx, ky, feq1, feq2):
    '''
    Integrates the Fermi-Dirac distribution to calculate the total carrier
    density (in m^-2). This is the sum of contributions from both valleys.

    Parameters:
    - kx, ky: Nkx, Nky arrays of kx, ky points
    - feq1, feq2: N(=4) x Nkx x Nky arrays of occupation for valley K and K'
    - EF: Fermi energy (eV)
    - T: temperature (K)
    '''
    return n_valley(kx, ky, feq1) + n_valley(kx, ky, feq2)


def disp_field(Delta, nt, nb):
    '''
    Returns the electric displacement field D/epsilon0 across bilayer graphene
    corresponding to an interlayer asymmetry Delta and carrier density on each
    layer nt and nb. We divide by the vacuum permittivity epsilon0 and return
    the displacement field in units of electric field (mV/nm)

    Parameters:
    - Delta: interlayer asymmetry (eV)
    - nt (nb): carrier density on top (bottom) layer (m^-2)

    Returns:
    - D: electric displacement field (?)
    '''
    D = Delta / d - q / eps0 * (nt - nb)  # note Delta in eV,
                                          # so we leave out e in denominator
    return D / 1e6  # V/m -> mV/nm


def _M_integral(kx, ky, feq, splE, splO, splM, tau=0, EF=0):
    '''
    Integrates over k space as part of the calculation for orbital magnetization
    for one valley and one band. Dotted with an applied electric field gives the
    magnetization for each band.

    Parameters:
    - kx, ky: Nkx, Nky arrays of kx, ky points
    - feq: Nkx x Nky array of equilibrium occupation
    - (splE, splO, splM) : splines for (energy / berry curvature / magnetic
        moment) for the given band
    - tau: scattering time (s). In general an Nkx x Nky array.
    - EF: Fermi energy (eV)

    Returns:
    - a length-2 array of magnetization "divided by" electric field
    '''
    E = splE(kx, ky)
    O = splO(kx, ky)
    Mu = splM(kx, ky)

    # non-equilibrium occupation ("divided by" dot product with E field)
    # equilibrium term will integrate to zero
    f = q * tau / hbar * np.array(np.gradient(feq, kx, ky, axis=(-2, -1)))

    integrand = 1 / (2 * np.pi) **2 * f * (Mu * muB + q * O / hbar * (EF - E))

    integral = simps(simps(integrand, ky, axis=-1), kx, axis=-1)
    return integral


def _M_integral_by_parts(kx, ky, feq, splE, splO, splM, tau=0, EF=0):
    '''
    Integrates over k space as part of the calculation for orbital magnetization
    for one valley and one band. Dotted with an applied electric field gives the
    magnetization for each band.

    This is the ''integration by parts'' version in which the derivatives
    are moved to the energy, magnetic moment, and berry curvature terms.

    Parameters:
    - kx, ky: Nkx, Nky arrays of kx, ky points
    - feq: Nkx x Nky array of equilibrium occupation
    - (splE, splO, splM) : splines for (energy / berry curvature / magnetic
        moment) for the given band
    - tau: scattering time (s). In general an Nkx x Nky array.
    - EF: Fermi energy (eV)

    Returns:
    - a length-2 array of magnetization "divided by" electric field
    '''
    E = splE(kx, ky)
    O = splO(kx, ky)
    Mu = splM(kx, ky)

    E_grad = np.array([splE(kx, ky, dx=1), splE(kx, ky, dy=1)])
    O_grad = np.array([splO(kx, ky, dx=1), splO(kx, ky, dy=1)])
    Mu_grad = np.array([splM(kx, ky, dx=1), splM(kx, ky, dy=1)])

    prefactor = - q * tau / hbar / (2 * np.pi) ** 2 * feq
    integrand = prefactor * (Mu_grad * muB \
                     + q / hbar * O_grad * (EF - E) \
                     - q / hbar * O * E_grad
    )

    integral = simps(simps(integrand, ky, axis=-1), kx, axis=-1)

    return integral


def _M_bands(kx, ky, feq, splE, splO, splM, tau=0, EF=0, byparts=True):
    '''
    Calculates the integral for orbital magnetization for each of four bands
    and sums over bands. Dotted with an applied electric field gives the
    total magnetization.

    Parameters:
    - kx, ky: Nkx, Nky arrays of kx, ky points
    - feq: N(=4) x Nkx x Nky array of equilibrium occupation
    - (splE, splO, splM) : N(=4) array of splines for (energy / berry curvature
        / magnetic moment) in each band
    - tau: scattering time (s). In general an Nkx x Nky array.
    - EF: Fermi energy (eV)
    - byparts: if True, will use the "integration by parts" version of the
        integral (function `_M_integral_by_parts`). If False, will use the
        function `_M_integral`

    Returns:
    - a length-2 array of x/y components of magnetization "divided by" E field
    '''
    N = feq.shape[0]
    M_no_dot_E = np.empty((N, 2))  # second dim is two components of integrand

    if byparts:
        integral = _M_integral_by_parts
    else:
        integral = _M_integral

    for i in range(N):
        M_no_dot_E[i] = integral(kx, ky, feq[i], splE[i], splO[i], splM[i],
                            tau=tau, EF=EF)

    return M_no_dot_E.sum(axis=0)  # sum over bands


def M_valley(kx, ky, feq, splE, splO, splM, Efield=[0,0], tau=0, EF=0):
    '''
    Integrates over k space to get orbital magnetization for one valley.

    Parameters:
    - kx, ky: Nkx, Nky arrays of kx, ky points
    - feq: N(=4) x Nkx x Nky array of equilibrium occupation
    - (splE, splO, splM) : N(=4) array of splines for (energy / berry curvature
        / magnetic moment) in each band
    - Efield: length-2 array of electric field x/y components (V/m)
    - tau: scattering time (s). In general an Nkx x Nky array.
    - EF: Fermi energy (eV)

    Returns:
    - 2D orbital magnetization (Ampere)
    '''
    return _M_bands(kx, ky, feq, splE, splO, splM, tau=tau, EF=EF).dot(Efield)


def D_valley(kx, ky, f, splO):
    '''
    Integrates over k space to get Berry curvature dipole for one valley.
    Integral is not summed over bands!

    Parameters:
    - kx, ky: Nkx, Nky arrays of kx, ky points
    - f: N(=4) x Nkx x Nky array of occupation for valley K or K'
    - splO: N(=4) array of splines for berry curvature in each band
    '''
    N = f.shape[0]  # 4
    D = np.empty(N)

    for n in range(N):
        Omega_dkx = splO[n](kx, ky, dx=1)
        D[n] = simps(simps(Omega_dkx * f[n], ky), kx)
            # integral over y (axis -1) then x (axis -1 of the result of simps)

    return D  # not yet summed over bands
