import numpy as np
from ase.atoms import Atoms
from SlabGenom import SlabGenom
from SlabModels import SlabBulk, Slab
from LatticeTools import convert_lattice_with_hkl_normal

BOHR_RADIUS_SI = 0.52917720859E-10 # m
BOHR_RADIUS_CM = BOHR_RADIUS_SI * 100.0
BOHR_RADIUS_ANGS = BOHR_RADIUS_CM * 1.0E8
PACK_THR: float = 1.0e-6     # internal coordinate
POSIT_THR = 0.01
STEP_FOR_GENOMS: float = 0.50  # angstromset()


def surfaces(lattice, indices, layers: int, vacuum: float=0.0, orthogonal=True, adsorbates=[]) -> Atoms:
    """Create surfaces from a given lattice and Miller indices.

    lattice: Atoms object or str
        Bulk lattice structure of alloy or pure metal. Both the 
        conventional cell and primitive cell can be used as the unit cell.
        One can also give the chemical symbol as a string, in which case the
        correct bulk lattice will be generated automatically.
    indices: sequence of three int
        Surface normal in Miller indices (h,k,l).
    layers: int
        Number of equivalent layers of the slab.
    vacuum: float
        Amount of vacuum added on both sides of the slab.
    orthogonal: bool
        Whether the lattice will be converted to that whose upper plane is 
        perpendicular to the normal vector specified by h, k, l
    adsorbates: adsorbates mapping list or str
        list of adsorbates mapping formatted as below:
        hydroxyl = Atoms()
        [
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
        adsorbate (necessary): str or Atom object or Atoms object
            Adsorbate atom (molecule) object or its symbol
        on (necessary): str
            Elemental symbol of the atom which the adsorbate will adsorb on.
        bond_length (optional): float
            Bond length between the adsorbate and the surface atom. If not specified,
            bond length of the dangling bond from the surface atom should have in the 
            bulk cell will be applied. .
        ads_atom_index (optional): int
            Index of the atom in Atoms object which will be bounded with the surface atom.
            If not specified, index 0 is used  by default.
        rotation_top (optional): tuple of three float
            Rotational angle (degree) around x, y, and z axes for the top-surface adsorbates. 
            The origin is the atom specified by `ads_atom_index`.
            The x, y, and z axes are global axes.
        rotation_bottom (optional): tuple of three float
            Rotational angle (degree) around x, y, and z axes for the top-surface adsorbates.
            The origin is the atom specified by `ads_atom_index`.
            The x, y, and z axes are global axes.
    """
    atoms = lattice
    atoms = convert_lattice_with_hkl_normal(atoms, *indices)
    slabs = get_slabs(atoms, layers)
    surfaces = [slab.to_atoms(adsorbates=adsorbates) for slab in slabs]
    for surface in surfaces:
        surface.center(vacuum=vacuum, axis=2)
    
    return surfaces


def get_slabs(lattice: SlabBulk, layers: int):
    """
    Generate a list of Slab objects from a given transformed lattice.
    Parameters:
        lattice (SlabBulk): A lattice object containing transformed bulk structure 
                            and metadata (e.g., transformation vectors).
        layers (int): Number of atomic layers to include in each slab.
    Returns:
        List[Slab]: List of generated slab models with different surface offsets.
    """

    trans_vec_set = lattice.trans_vec_set
    nstep = int(trans_vec_set[2][2] / STEP_FOR_GENOMS)
    slab_genoms = {}
    for i in range(nstep):
        offset = i / nstep
        slab_genom = get_slab_genom(offset, trans_vec_set, lattice.atoms)
        if slab_genom is not None and slab_genom not in slab_genoms:
            slab_genoms[slab_genom] = offset

    offsets = list(slab_genoms.values())
    lattice.setup_bonds()
    slabs = [lattice.to_slab(offset, layers) for offset in offsets]

    return slabs


def get_slab_genom(offset: float, trans_vec_set: np.ndarray, atoms: Atoms):
    """
    Generate a unique "genome" for a slab structure at a given z-offset.

    The genome represents the vertical arrangement of atoms, used to detect 
    structurally unique slabs (e.g., for different terminations or shifts).

    Parameters:
        offset (float): Fractional shift along the z-direction (0.0 to <1.0).
        trans_vec_set (np.ndarray): 3x3 array of transformed lattice vectors.
        atoms (ASE_Atoms or similar): Atomic structure to analyze.

    Returns:
        SlabGenom: An object representing atomic species and their z-ordering.
    """

    names = []
    coords = []

    slab_thickness = trans_vec_set[2][2]
    iatom = len(atoms)

    for idx, atom in enumerate(atoms):
        # Compute fractional z-coordinate with offset
        z_frac = atom.get_coord_frac(trans_vec_set)[2] + offset
        z_frac_wrapped = z_frac % 1.0

        # Correct atoms near the upper boundary (to wrap into lower side)
        if (1.0 - z_frac_wrapped) * slab_thickness < POSIT_THR:
            z_frac_wrapped -= 1.0

        # Identify the "first" atom just below or at the z=0 surface
        if iatom >= len(atoms):
            dz = abs(z_frac - z_frac_wrapped) * slab_thickness
            if dz < POSIT_THR:
                iatom = idx

        names.append(atom.element)
        coords.append(z_frac_wrapped * slab_thickness)

    # Rotate atom list so that atom at iatom comes first
    names_rot = names[iatom:] + names[:iatom]
    coords_rot = coords[iatom:] + coords[:iatom]

    return SlabGenom(names_rot, coords_rot)