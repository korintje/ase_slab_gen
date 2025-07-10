from SlabModelStem import SlabModelStem
from ase import Atoms
from ase.build import surface, bulk, add_vacuum
from ase.io import write, read
import matplotlib.pyplot as plt
from ase.visualize.plot import plot_atoms
import os
from general_surfaces import surfaces
# from LatticeTools import convert_by_hkl


# Save slab image with surface extension
def save_slab_image(slab, filename):
    ext_slab = slab * (4, 4, 1)
    # ext_slab = slab
    fig, axarr = plt.subplots(1, 4, figsize=(20, 5), dpi=300)
    plot_atoms(ext_slab, axarr[0], rotation=('0x,0y,0z'))
    plot_atoms(ext_slab, axarr[1], rotation=('-90x,0y,0z'))
    plot_atoms(ext_slab, axarr[2], rotation=('-90x,-90y,0z'))
    plot_atoms(ext_slab, axarr[3], rotation=('30x,0y,0z'))
    fig.savefig(filename)

material_id = "mp-3901"

cif_filename = f"{material_id}.cif"

# Get and save structure from material id
if os.path.isfile(cif_filename):
    mybulk = read(cif_filename)
    # bulk = parser.get_structures()[0]
    # print(f"Local cif file: {cif_filename} has been loaded")
else:
    mybulk = bulk("OZn", crystalstructure="wurtzite", a=3.289, b=3.289, c=5.307, alpha=90.000, u=None)
    # mybulk = mybulk.repeat((3, 3, 2))

cell = mybulk
h = 1
k = 0
l = 0

hydroxyl = Atoms('OH', positions=[[0, 0, 0], [0.96, 0, 0]])
ads = [
    {"adsorbate": "H", "on": "O", "bond_length": 1.0},
    {
        "adsorbate": hydroxyl, 
        "on": "Zn", 
        "bond_length": 1.5, 
        "ads_atom_index": 0,
        "rotation_top": (90.0, 45.0, 0.0),
        "rotation_bottom": (0.0, 45.0, 0.0),
    },            
]

# slab_model_stem = SlabModelStem(cell, h, k, l)
# slab_models = slab_model_stem.get_slab_models()
slabs = surfaces(cell, (h, k, l), 1, vacuum=10.0, adsorbates=ads)
for i, slab in enumerate(slabs):
#     slab_model.set_thickness(4.0)
    # print(slab_model.__dict__)
    # slab_model.set_vacuum(0.0)
    # slab_model.set_scaleA(1)
    # slab_model.set_scaleB(1)
    # slab_cell = slab_model.to_atoms()
    # print(slab)
    # print(slab_cell.get_dipole_moment())
    # add_vacuum(slab_cell, 30.0)
    save_slab_image(slab, f"{i}.png")


"""
converted_cell = convert_by_hkl(cell, h , k, l)
slabs = get_slabs(converted_cell, thickness=1, vacuum=0.0, adsorbates=None)
for slab in slabs:
    pass
"""

# myslab = surface(mybulk, (0, 1, 0), 3)
# myslab = myslab.repeat((3,4,1))