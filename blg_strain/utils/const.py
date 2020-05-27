import numpy as np

# Physical parameters
hbar = 6.5821e-16 # eV * s
q = 1.602176634e-19 # Coulomb, electron charge (also eV to J conversion)
m_e = 9.1093837015e-31 / q # kg divide by electron charge to take care of a J-> eV conversion
muB = q * hbar / (2 * m_e) # Bohr magneton J/T = A*m^2 (hbar/m takes care of J->eV conversion)
kB = 8.617333262145e-5 # Boltzmann constant eV/K
eps0 = 8.8541878128e-12 # vacuum permittivity C/V*m

# Graphene
a = 0.246e-9 # meters, lattice constant for graphene
d = 0.34e-9 # meters, interlayer distance for BLG
A_BZ = 2 * np.sqrt(3) * (np.pi / a) ** 2  # area of Brillouin zone, m^-2

nu = 0.165  # Poisson ratio for graphene (in general this should be that of the substrate)

# Hopping parameters (eV)
gamma0 = 3.161
gamma1 = 0.381
gamma3 = 0.38
gamma4 = 0.14 * 0
dab = 0.022 * 0 # dimer asymmetry

# Fermi velocities m * eV / (eV * s) = m/s
v0 = np.sqrt(3) * a * gamma0 / (2 * hbar)
v3 = np.sqrt(3) * a * gamma3 / (2 * hbar)
v4 = np.sqrt(3) * a * gamma4 / (2 * hbar)

# Estimated Gruneisen parameters
# eta0 = -2
eta0 = -3  # less consistent with experiments but matches theory papers
eta3 = -1
eta4 = -1
