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
slabs = surfaces(zno_bulk, (h, k, l), 4, vacuum=5.0, adsorbates=ads)

# Save slab images
fig, axarr = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
for i, slab in enumerate(slabs):
    ext_slab = slab * (4, 4, 1)
    plot_atoms(ext_slab, axarr[i], rotation=('-90x,-90y,0z'))
fig.savefig(f"ZnO_100.png")
