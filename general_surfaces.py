# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 Takuro Hosomi
# Email: t.hosomi1410@gmail.com
# GitHub: https://github.com/korintje
#
# This script is licensed under the GNU Lesser General Public License v2.1 (LGPL-2.1).
# You should have received a copy of the GNU Lesser General Public License along with this script.
# If not, see <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>.
#
# The core slab generation logic is originally developed by nisihara.burai@gmail.com.
# GitHub: https://github.com/BURAI-team/burai


import numpy as np
from ase.atoms import Atoms
from SlabGenom import SlabGenom
from SlabModels import SlabBulk, Slab
from LatticeTools import convert_lattice_with_hkl_normal


# Thresholds and steps
POSITION_THRESHOLD = 0.01    # Tolerance for detecting atomic position jumps (in Å)
SLAB_STEP_SIZE: float = 0.50 # Step size for slab genome generation (in Å)


def surfaces(
    lattice: Atoms,
    miller_indices: tuple[int, int, int],
    num_layers: int,
    vacuum: float = 0.0,
    orthogonal: bool = True,
    adsorbates: list = []
) -> list[Atoms]:
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
    adsorbates: list of adsorbates mapping formatted as below:
        adsorbate (necessary): str or Atom object or Atoms object
            Adsorbate atom (molecule) object or its symbol
        on (necessary): str
            Elemental symbol of the atom which the adsorbate will adsorb on.
        bond_length (optional): float
            Bond length between the adsorbate and the surface atom. If not specified,
            bond length of the dangling bond from the surface atom should have in the 
            bulk cell will be applied.
        ads_atom_index (optional): int
            Index of the atom in Atoms object which will be bounded with the surface atom.
            If not specified, index 0 is used by default.
    """
    # Transform lattice using the Miller indices
    transformed_lattice = convert_lattice_with_hkl_normal(lattice, *miller_indices)

    # Generate slab configurations
    slabs = _generate_slabs(transformed_lattice, num_layers)

    # Convert slab objects to ASE Atoms
    surfaces = []
    for slab in slabs:
        surface = slab.to_atoms(adsorbates=adsorbates)
        if orthogonal:
            _make_cell_orthogonal(surface)
        surface.center(vacuum=vacuum, axis=2)
        surfaces.append(surface)

    return surfaces


def _generate_slabs(bulk: SlabBulk, num_layers: int) -> list[Slab]:
    """
    Generate unique slab configurations.
    bulk: SlabBulk
        Transformed lattice.
    num_layers: int
        Number of atomic layers in slab.
    Returns
    -------
    list of Slab
        Unique slab configurations.
    """
    slab_thickness = bulk.trans_vec_set[2][2]
    num_steps = int(slab_thickness / SLAB_STEP_SIZE)
    unique_genomes = {}

    for i in range(num_steps):
        offset = i / num_steps
        genome = _get_slab_genome(offset, bulk.trans_vec_set, bulk.atoms)

        # Add only unique genomes
        if genome is not None and genome not in unique_genomes:
            unique_genomes[genome] = offset

    bulk.setup_bonds()
    return [bulk.to_slab(offset, num_layers) for offset in unique_genomes.values()]


def _get_slab_genome(offset: float, cell_vectors: np.ndarray, atoms: Atoms) -> SlabGenom | None:
    """
    Generate a unique representation (SlabGenom) of a slab by projecting atoms
    along the surface-normal direction and sorting them.
    offset: float
        Fractional offset along the surface normal.
    cell_vectors : np.ndarray
        Lattice vectors defining the slab coordinate system.
    atoms: Atoms
        Atomic structure.
    """
    names = []
    z_coords = []
    slab_thickness = cell_vectors[2][2]
    rotate_index = len(atoms)

    for idx, atom in enumerate(atoms):
        # Get fractional z-coordinate with offset and wrap to [0, 1)
        z_frac = atom.get_coord_frac(cell_vectors)[2] + offset
        z_frac_wrapped = z_frac % 1.0

        # If atom is close to the top boundary, bring it back into the slab
        if (1.0 - z_frac_wrapped) * slab_thickness < POSITION_THRESHOLD:
            z_frac_wrapped -= 1.0

        # Select a rotation point to reorder atoms if needed
        if rotate_index >= len(atoms):
            dz = abs(z_frac - z_frac_wrapped) * slab_thickness
            if dz < POSITION_THRESHOLD:
                rotate_index = idx

        names.append(atom.element)
        z_coords.append(z_frac_wrapped * slab_thickness)

    # Rotate atom list so the same atoms appear in the same order
    names_rot = names[rotate_index:] + names[:rotate_index]
    z_coords_rot = z_coords[rotate_index:] + z_coords[:rotate_index]

    return SlabGenom(names_rot, z_coords_rot)


def _make_cell_orthogonal(atoms: Atoms) -> None:
    """
    Modify the cell so that the c-vector is perpendicular to the other vectors.
    atoms: ASE Atoms object with current cell.
    """
    a_vec, b_vec, c_vec = atoms.get_cell()

    # Calculate new c-vector perpendicular to a and b
    new_c_dir = np.cross(a_vec, b_vec)
    norm = np.linalg.norm(new_c_dir)

    if norm < 1e-8:
        raise ValueError("Cell vectors a and b are colinear.")

    new_c_unit = new_c_dir / norm
    proj_length = np.dot(c_vec, new_c_unit)
    new_c_vec = new_c_unit * proj_length

    # Apply updated orthogonal cell
    new_cell = np.array([a_vec, b_vec, new_c_vec])
    atoms.set_cell(new_cell, scale_atoms=False)
    atoms.set_scaled_positions(atoms.get_scaled_positions())
