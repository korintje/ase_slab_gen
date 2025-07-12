# ASE Slab Generator (Improved Version)

This repository provides an enhanced tool for generating crystal slabs for atomic simulations, improving upon the standard slab generation features available in the [Atomic Simulation Environment (ASE)](https://wiki.fysik.dtu.dk/ase/).

## Overview

ASE (Atomic Simulation Environment) provides built-in tools for slab generation, but they come with several limitations.　First, while conventional unit cells can be used as the source for slab construction, primitive cells are not supported.　Second, although multiple slab structures can theoretically exist for a given cleavage plane specified by Miller indices (hkl), the standard ASE slab generator always returns only a single structure. As a result, some desirable or stable slab configurations may not be accessible.

This script set addresses these limitations by offering a more general and exhaustive approach to slab generation. It is based on the “SlabGenom” algorithm developed by Satomichi Nishihara, which is also used in the Java-based simulation platform [BURAI](https://github.com/BURAI-team/burai).

In addition, ASE’s default tools make it difficult to handle adsorbates except in basic cases. In contrast, this script set employs original logic that allows for more flexible and precise design of adsorption structures, even for complex surface terminations or custom adsorbate placements.

## Limitations of Standard ASE Slab Tools

Although ASE provides basic tools for generating crystal slabs, users may encounter several issues:

- **Incomplete Surface Enumeration**: ASE may not generate all symmetry-inequivalent surface terminations for a given Miller index.
- **Cannot use primitive cells**: ASE's surface() function can only use conventional cells, not primitive cells.
- **Unstable or Unphysical Slabs**: The resulting slabs may sometimes be chemically or structurally unstable.
- **Manual Adsorbate Placement**: Accurately placing adsorbates on arbitrary surfaces often requires tedious manual adjustments, especially for non-standard structures.

## What This Tool Offers

This improved slab generator addresses the above issues by:

- **Systematically Enumerating All Terminations**: For a given Miller index, the tool generates all symmetry-inequivalent slab terminations that are physically meaningful.
- **Primitive cells available**: Both the conventional cells and primitive cells can be used in the `surfaces()` function.
- **Handling Complex Structures**: Robust support for arbitrary bulk structures beyond simple fcc, bcc, or hcp types.
- **Flexible Adsorbate Support**: Allows better control over adsorbate positioning.

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
The core slab generation logic is adapted from the SlabGenom system developed by nisihara.burai@gmail.com. It is also implemented in the [BURAI](https://github.com/BURAI-team/burai) graphical interface for quantum chemical calculations.
