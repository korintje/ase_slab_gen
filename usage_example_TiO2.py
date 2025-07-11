from ase.io import write
import matplotlib.pyplot as plt
from ase.visualize.plot import plot_atoms
from general_surfaces import surfaces
from ase.spacegroup import crystal

a = 4.6
c = 2.95

# Rutile TiO2:
my_bulk = crystal(
    ['Ti', 'O'],
    basis=[(0, 0, 0), (0.3, 0.3, 0.0)],
    spacegroup=136,
    cellpar=[a, a, c, 90, 90, 90],
)

# Set hkl indices
h, k, l = 0, 0, 1

# Create all possible slabs
slabs = surfaces(my_bulk, (h, k, l), 4, vacuum=10.0, adsorbates=[], orthogonal=True)

# Save slab images
for i, slab in enumerate(slabs):
    ext_slab = slab * (4, 4, 1)
    fig, axarr = plt.subplots(1, 4, figsize=(20, 5), dpi=300)
    plot_atoms(ext_slab, axarr[0], rotation=('0x,0y,0z'))
    plot_atoms(ext_slab, axarr[1], rotation=('-90x,0y,0z'))
    plot_atoms(ext_slab, axarr[2], rotation=('-90x,-90y,0z'))
    plot_atoms(ext_slab, axarr[3], rotation=('30x,0y,0z'))
    write(f"{i}.cif", slab)
    fig.savefig(f"{i}.png")
