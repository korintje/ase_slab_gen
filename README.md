# ASE Slab Generator (Improved Version)

This repository provides an enhanced tool for generating crystal slabs for atomic simulations, improving upon the standard slab generation features available in the [Atomic Simulation Environment (ASE)](https://wiki.fysik.dtu.dk/ase/).

## Limitations of Standard ASE Slab Tool

Although ASE provides basic function `surface()` for generating crystal slabs, users may encounter several issues:

- **Cannot use primitive cells**: `surface()` can only use conventional cells, not primitive cells.
- **Incomplete Surface Enumeration**: Although multiple slab structures can theoretically exist for a given cleavage plane specified by Miller indices (hkl), the `surface()` always returns only a single structure.
- **Manual Adsorbate Placement**: Accurately placing adsorbates on arbitrary surfaces often requires tedious manual adjustments, excepting some special structures.

## What This Tool Offers

This improved slab generator `surfaces()` in `general_surfaces.py` addresses the above issues by:

- **Primitive cells available**: Both the conventional cells and primitive cells can be used.
- **Systematically Enumerating All Terminations**: A general and exhaustive approach was used to slab generations based on “SlabGenom” algorithm originally developed by Satomichi Nishihara. He used this algorithm in a Java-based application [BURAI](https://github.com/BURAI-team/burai).
- **Flexible Adsorbate Support**: Allows general controls over adsorbate positioning by considering "dangling bonds" created when cutting the bulk crystal.

e.g.) ZnO(100)

<img width="1200" height="600" alt="ZnO_100" src="https://github.com/user-attachments/assets/2d369f5f-eb58-4208-bd5e-5010cf9d75b4" />

## Getting Started

### Installation

Clone the repository using:

```bash
git clone https://github.com/korintje/ase_slab_gen.git
```

Make sure you have ASE installed (e.g., via pip install ase).

### Basic Usage

Here's a minimal example to generate all symmetry-unique (111) slabs of silicon:

After cloning, you can import the surfaces function from the general_surfaces module in your Python script to begin generating slabs.
Here is a basic example demonstrating how to generate slabs for a silicon crystal:

```python
from ase import Atoms
from ase.build import bulk
import matplotlib.pyplot as plt
from ase.visualize.plot import plot_atoms
from general_surfaces import surfaces

# Create a bulk
zno_bulk = bulk("OZn", crystalstructure="wurtzite", a=3.289, b=3.289, c=5.307, alpha=90.000, u=None)

# Set hkl indices
h, k, l = 1, 0, 0

# Define adsorbates
hydroxyl = Atoms('OH', positions=[[0, 0, 0], [0, 0, 0.96]])
ads = [
    {"adsorbate": "H", "on": "O", "bond_length": 1.0},
    {
        "adsorbate": hydroxyl, 
        "on": "Zn", 
        "bond_length": 1.5, 
        "ads_atom_index": 0,
    },            
]

# Create all possible slabs (e.g. Two surfaces for ZnO(100))
slabs = surfaces(zno_bulk, (h, k, l), 4, vacuum=10.0, adsorbates=ads)

# Save slab images
fig, axarr = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
for i, slab in enumerate(slabs):
    ext_slab = slab * (4, 4, 1)
    plot_atoms(ext_slab, axarr[i], rotation=('-90x,-90y,0z'))
fig.savefig(f"ZnO_100.png")
```
> Optional arguments allow fine-tuning slab thickness, symmetry tolerances, and more. See function docstrings for full details.

# Acknowledgments
The slab generation logic is based on “SlabGenom” algorithm originally developed by Satomichi Nishihara. It is also implemented in the [BURAI](https://github.com/BURAI-team/burai) graphical interface for quantum chemical calculations.
