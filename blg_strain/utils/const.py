import numpy as np

# Physical parameters
q = 1.602176634e-19 # Coulomb, electron charge (also eV to J conversion)
hbar = 6.5821e-16 # eV * s
hbar_J = hbar * q  # J * s
m_e = 9.1093837015e-31 / q # kg divide by electron charge to take care of a J-> eV conversion
muB = q * hbar / (2 * m_e) # Bohr magneton J/T = A*m^2 (hbar/m takes care of J->eV conversion)
kB = 8.617333262145e-5 # Boltzmann constant eV/K
eps0 = 8.8541878128e-12 # vacuum permittivity C/V*m
mu0 = 4 * np.pi * 1e-7 # vacuum permeability N/A^2

# Graphene
a0 = 0.142e-9  # meters, C-C bond length - careful not to confuse for lattice const!
a = np.sqrt(3) * a0  # meters, lattice constant for graphene
d = 0.34e-9  # meters, interlayer distance for BLG
A_BZ = 2 * np.sqrt(3) * (np.pi / a) ** 2  # area of Brillouin zone, m^-2
K = 4 * np.pi / (3 * np.sqrt(3))  # kx coordinate of K valley (no strain)
nu = 0.165  # Poisson ratio for graphene (in general this should be that of the substrate)

# Hopping parameters (eV)
gamma0 = 3.16
gamma1 = 0.381
gamma3 = 0.38
gamma4 = 0.14 * 0
gamman = 0.1 * gamma0 * 0
DeltaAB = 0.022 * 0  # dimer asymmetry

# Estimated Gruneisen parameters
# eta0 = -2
eta0 = -3  # less consistent with experiments but matches theory papers
eta3 = -1
eta4 = -1
etan = -1

# Hopping bonds - in units of the atomic separation a0
# Nearest neighbor
delta1 = np.array([0, 1])
delta2 = np.array([np.sqrt(3)/2, -1/2])
delta3 = np.array([-np.sqrt(3)/2, -1/2])
deltas = [delta1, delta2, delta3]

# Next-nearest neighbor
deltan1 = np.array([np.sqrt(3), 0])
deltan2 = np.array([-np.sqrt(3), 0])
deltan3 = np.array([np.sqrt(3)/2, 3/2])
deltan4 = np.array([np.sqrt(3)/2, -3/2])
deltan5 = np.array([-np.sqrt(3)/2, 3/2])
deltan6 = np.array([-np.sqrt(3)/2, -3/2])
deltans = [deltan1, deltan2, deltan3, deltan4, deltan5, deltan6]
