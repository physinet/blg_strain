# blg_strain
Band structure, Berry curvature, and orbital magnetic moment calculations for strained bilayer graphene.

## Installing:
Download this repository, navigate to the directory in a command prompt, and run
`pip install -e .` to install the module and enable editing.

## Example:
The following code performs calculations for 2% uniaxial strain and an interlayer asymmetry Δ = 30 meV.
```python
from blg_strain.bands import get_bands
from blg_strain.utils.plotting import plot_bands_KKprime

Nkx, Nky = 200, 200  # number of points in k space
Delta = 0.02  # interlayer asymmetry, eV
delta = 0.02  # uniaxial strain

kx, ky, Kx, Ky, E, Psi, Omega, Mu = get_bands(Nkx=Nkx, Nky=Nky, xi=1, Delta=Delta, delta=delta)
kx, ky, Kx, Ky, E1, Psi1, Omega1, Mu1 = get_bands(Nkx=Nkx, Nky=Nky, xi=-1, Delta=Delta, delta=delta)

fig, ax = plot_bands_KKprime(Kx, Ky, E, E1)
fig, ax = plot_bands_KKprime(Kx, Ky, Omega, Omega1)
fig, ax = plot_bands_KKprime(Kx, Ky, Mu, Mu1)
```
(https://github.com/physinet/blg_strain/blob/master/plots/E_Delta20meV_delta2.png)
(https://github.com/physinet/blg_strain/blob/master/plots/Omega_Delta20meV_delta2.png)
(https://github.com/physinet/blg_strain/blob/master/plots/Mu_Delta20meV_delta2.png)
