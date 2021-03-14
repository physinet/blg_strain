# blg_strain - calculations for magnetoelectric effect in strained bilayer graphene
This is a Python package used to perform the calculations described in the article ["Electrically tunable and reversible magnetoelectric coupling in strained bilayer graphene
" (arXiv:2103.04124)](https://arxiv.org/abs/2103.04124).

## Installation
Download this repository, navigate to the directory in a command prompt, and run
`pip install -e .` to install the module and enable editing.

## Quickstart
Below is a description of the basic usage of this package, lacking however explicit details about the strained bilayer graphene (sBLG) model.
For a detailed explanation of the model, please see the accompanying article.
Please also see the [`example.ipynb`](https://github.com/physinet/blg_strain/blob/master/example.ipynb) notebook, which has the below commands and accompanying plots of the calculated quantities.

### Define the lattice
The `StrainedLattice` class stores information about the sBLG lattice, where the strain is parameterized by the variables `eps` (strain magnitude) and `theta` (strain angle relative to the $x$ axis).
Individual hopping parameters can be turned off using the `turn_off` keyword argument.
```python
from blg_strain.lattice import StrainedLattice

sl = StrainedLattice(eps=0.01, theta=0)  # 1% uniaxial tensile strain along x
sl.calculate(turn_off=['gamma3', 'gamma4'])  # calculate with gamma3 and gamma4 hopping turned off
```
The most relevant calculated attributes of the `sl` object are:
- `sl.K` and `sl.Kp`: locations of the K and K' points in momentum space
- `sl.gamma0s`, `sl.deltas`, etc.: modified hopping parameters and bond vectors for the strained lattice

### Calculate the band structure
The `BandStructure` class stores the energy, wavefunctions, Berry curvature, and orbital magnetic moment for sBLG calculated over a window of momentum space surrounding the K point.
The quantities at the K' point can be calculated considering the relevant symmetry operations (see article for discussion).
The `BandStructure` class takes as arguments:
- `sl`, an instance of the `StrainedLattice` class
- `window`, width of the window of sampled momentum space
- `Delta`, the interlayer asymmetry

```python
from blg_strain.bands import BandStructure

bs = BandStructure(sl=sl, window=0.1, Delta=0.01)
bs.calculate(Nkx=200, Nky=200)  # 200x200 grid
```
The most relevant calculated attributes of the `bs` object are:
- `bs.kxa` (length `Nkx`), `bs.kya` (length `Nky`): arrays defining the momentum-space grid surrounding the K valley.
- `bs.Kxa`, `bs.Kya` (`Nkx x Nky`): ['ij' indexed meshgrid](https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html) made from `bs.kxa` and `bs.kya`
- `bs.E`, `bs.Omega`, `bs.Mu` (`4 x Nkx x Nky`): energy eigenvalues, Berry curvature, and orbital magnetic moment for each of 4 bands at each coordinate in momentum space. Bands indexed 1 and 2 are the valence and conduction bands, respectively.
- `bs.Psi` (`4 x 4 x Nkx x Nky`): eigenstates for each of 4 bands (dimension 0) with 4 components (dimension 1) at each coordinate in momentum space
- `bs.splE`, etc.: length-4 array of bivariate splines for each quantity, allowing for interpolation onto a finer grid. `bs.splE[2](kxa, kya)` returns a two-dimensional array of conduction band energies calculated over a grid defined by the one-dimensional arrays `kxa` and `kya`.

### Calculate properties of bands with filled states
The `FilledBands` class stores the carrier density, displacement field, and magnetoelectric coefficient for a band structure filled up to a given Fermi level `EF` (`EF=0` defined as the middle of the band gap).
These quantities are computed by integrating over the `window` defined in the `BandStructure` instance `bs`, which can describe the entire Brillouin zone as long as the Fermi surface lies completely within the window.
```python
from blg_strain.bands import FilledBands

fb = FilledBands(bs=bs, EF=0.01)
fb.calculate(Nkx=500, Nky=500)  # resampled to 500x500 grid
```
The most relevant calculated attributes of the `fb` object are:
- `fb.alpha`: $x$ and $y$ components of linear magnetoelectric susceptibility
- `fb.D`: electric displacement field
- `fb.n`: carrier density

### Saving and loading results
Each of the above classes inherits from `blg_strain.utils.saver.Saver`, which enables straightforward saving and loading to the [HDF5 file format](https://www.h5py.org/).
Because of the hierarchical nature of the three classes (`FilledBands` depends on `BandStructure` depends on `StrainedLattice`), `BandStructure` objects are saved in a subdirectory of the directory where the corresponding `StrainedLattice` object is saved, and `FilledBands` objects are saved in a subdirectory of the directory where the corresponding `BandStructure` object is saved.
The following code can be used to save the objects created in the examples above:
```python
sl.save('example')
bs.save()
fb.save()
```
This creates the following file structure:
```
/
└───example
    │   StrainedLattice_eps0.010_theta0.000_Run0.h5   
    └───StrainedLattice_eps0.010_theta0.000_Run0
        │   BandStructure_Nkx200_Nky200_Delta10.000.h5
        └───BandStructure_Nkx200_Nky200_Delta10
            │   FilledBands_Nkx500_Nky500_EF15.000.h5
```
The objects can be reloaded at any time using (for example) `sl = StrainedLattice.load(filename)`.

It is often useful to create files consisting of only the final derived quantities resulting from a series of `BandStructure` and `FilledBands` calculations.
Loading such a "summary" file can take considerably less time than loading the individual `FilledBands` files.
The function `blg_strain.utils.saver.load` creates a summary file in the `StrainedLattice` directory.
Subsequently calling the `load` function will load from `summary.h5` instead of re-making the summary file:
```python
from blg_strain.utils.saver import load
Deltas, EFs, ns, Ds, alphas = load(sl_path)
```

## Project structure
The `blg_strain` package contains classes and methods used in calculating the electronic properties of strained bilayer graphene (sBLG).
The `plots` directory contains a jupyter notebook used to generate figures for the paper from the simulation results in `plots/data`.
This directory also contains final PDF files for each of the figures (some of which have been edited in Adobe Illustrator).

Briefly, the files under `blg_strain` are responsible for the following:
- `bands.py`: contains classes to contain results of band structure calculations for combinations of parameters
- `berry.py`: calculates Berry curvature and orbital magnetic moment from the eigenstates of the Hamiltonian
- `hamiltonian.py`: defines the $4\times4$ tight-binding Hamiltonian
- `lattice.py`: contains all calculations related to the geometry of the sBLG lattice
- `macroscopic.py`: calculations for "macroscopic" quantities: carrier density, electric displacement field, magnetoelectric coefficient (an integral of the Berry curvature and orbital magnetic moment over momentum space)
- `microscopic.py`: calculations for "microscopic" quantities, namely the Fermi-Dirac distribution
- `strayfield.py`: calculations for the stray field from finite-width/finite-length current-carrying wires and a uniformly magnetized rectangle. These calculations are used to imagine how to best study the magnetoelectric effect in sBLG experimentally.
- Utilities (`/utils`)
  - `const.py`: physical constants and values for hopping parameters, etc. for BLG
  - `plotting.py`: helper functions to plot band structure
  - `saver.py`: defines a class to aid with saving and loading calculation results
  - `utils.py`: miscellaneous utilities, mostly used for defining the calculation grid


## Contributing
Pull requests are welcome. Please [contact the author](mailto:physinet@gmail.com) to discuss potential applications of this code towards other computational studies.

## License
[MIT](https://choosealicense.com/licenses/mit/)
