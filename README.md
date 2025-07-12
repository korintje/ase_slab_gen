# ASE Slab Generator (Improved Version)

This repository provides an enhanced tool for generating crystal slabs for atomic simulations, improving upon the standard slab generation features available in the [Atomic Simulation Environment (ASE)](https://wiki.fysik.dtu.dk/ase/).

In contrast, 

## Limitations of Standard ASE Slab Tools

Although ASE provides basic tools for generating crystal slabs, users may encounter several issues:

- **Cannot use primitive cells**: ASE's `surface()` function can only use conventional cells, not primitive cells.
- **Incomplete Surface Enumeration**: Although multiple slab structures can theoretically exist for a given cleavage plane specified by Miller indices (hkl), the standard ASE slab generator always returns only a single structure.
- **Manual Adsorbate Placement**: Accurately placing adsorbates on arbitrary surfaces often requires tedious manual adjustments, excepting some special structures.

## What This Tool Offers

This improved slab generator `surfaces()` in `general_surfaces.py` addresses the above issues by:

- **Primitive cells available**: Both the conventional cells and primitive cells can be used.
- **Systematically Enumerating All Terminations**: A general and exhaustive approach was used to slab generations based on “SlabGenom” algorithm originally developed by Satomichi Nishihara. He used this algorithm in a Java-based application [BURAI](https://github.com/BURAI-team/burai).
- **Flexible Adsorbate Support**: Allows general controls over adsorbate positioning by considering "dangling bonds" created when cutting the bulk crystal.

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
import sys
sys.path.append('path/to/ase_slab_gen')  # e.g. './ase_slab_gen'

from ase.build import bulk
from general_surfaces import surfaces

# 1. Create a bulk silicon crystal
si = bulk('Si', 'diamond', a=5.43)

# 2. Generate slabs for the (1, 1, 1) plane with 10 Å slab and 10 Å vacuum
surface_list = surfaces(si, (1, 1, 1), slab_thick=10, vacuum_thick=10)

# 3. Save each slab to file
for i, slab in enumerate(surface_list):
    slab.write(f'Si_111_slab_{i}.xyz')

print(f"Generated {len(surface_list)} unique slab(s).")
```
> Optional arguments allow fine-tuning slab thickness, symmetry tolerances, and more. See function docstrings for full details.

# Acknowledgments
The slab generation logic is based on “SlabGenom” algorithm originally developed by Satomichi Nishihara. It is also implemented in the [BURAI](https://github.com/BURAI-team/burai) graphical interface for quantum chemical calculations.
