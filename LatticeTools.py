# -*- coding: utf-8 -*-

import numpy as np
from ase import Atoms
from SlabModels import SlabAtom, SlabBulk
from math import lcm

PACK_THR: float = 1.0e-6
POSIT_THR = 0.01


def orient_lattice_xy(lattice_vecs: np.ndarray) -> np.ndarray:
    """
    Rotate the lattice vectors with following rules:
    The first lattice vector is aligned with the x-axis. 
    The second lattice vector lies in the x-y plane.
    """
    a, b, c = np.linalg.norm(lattice_vecs, axis=1)
    cos_alpha = np.dot(lattice_vecs[1], lattice_vecs[2]) / (b * c) if b * c > 0 else 1.0
    cos_beta  = np.dot(lattice_vecs[0], lattice_vecs[2]) / (a * c) if a * c > 0 else 1.0
    cos_gamma = np.dot(lattice_vecs[0], lattice_vecs[1]) / (a * b) if a * b > 0 else 1.0
    
    cos_alpha_sq = cos_alpha ** 2
    cos_beta_sq  = cos_beta  ** 2
    cos_gamma_sq = cos_gamma ** 2
    sin_gamma_sq = 1.0 - cos_gamma_sq
    sin_gamma = np.sqrt(sin_gamma_sq)

    val = 1.0 + 2.0 * cos_alpha * cos_beta * cos_gamma - (cos_alpha_sq + cos_beta_sq + cos_gamma_sq)
    term2 = np.sqrt(val) / sin_gamma

    new_lattice_vecs = np.zeros((3, 3), dtype=float)
    new_lattice_vecs[0] = [a, 0.0, 0.0]
    new_lattice_vecs[1] = [b * cos_gamma, b * sin_gamma, 0.0]
    new_lattice_vecs[2] = [
        c * cos_beta,
        c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma,
        c * term2
    ]

    return new_lattice_vecs


def get_intercepts(h: int, k: int, l: int):
    """
    Given Miller indices (h, k, l), return:
    - Intercepts with each axis (in lattice units)
    """

    miller = np.array([h, k, l])
    nonzero = miller[miller != 0]

    if nonzero.size == 0:
        intercepts = np.zeros(3, dtype=int)
    else:
        scale = lcm(*nonzero)
        intercepts = np.array([
            scale // v if v != 0 else 0
            for v in miller
        ])

    return intercepts


def get_basis_trans_matrix(intercepts: np.ndarray) -> np.ndarray:
    """
    Construct a basis transformation matrix for lattice transformation,
    aligning the given Miller plane (defined by its intercepts) as the new c-axis.
    Parameters
    ----------
    intercepts : np.ndarray
        Integer intercepts of the Miller plane with the crystal axes.
    Returns
    -------
    np.ndarray
        A 3x3 integer matrix:
        - The first two rows span the Miller plane,
        - The third row is the Miller plane normal vector.
    """
    vectors = np.zeros((3, 3), dtype=int)
    signs = np.sign(intercepts)
    count = np.count_nonzero(intercepts)

    if count <= 1:
        # One-axis intercept
        if intercepts[0] != 0:
            if intercepts[0] > 0:
                vectors[0, 1] = 1
                vectors[1, 2] = 1
                vectors[2, 0] = 1
            else:
                vectors[0, 2] = 1
                vectors[1, 1] = 1
                vectors[2, 0] = -1

        elif intercepts[1] != 0:
            if intercepts[1] > 0:
                vectors[0, 2] = 1
                vectors[1, 0] = 1
                vectors[2, 1] = 1
            else:
                vectors[0, 0] = 1
                vectors[1, 2] = 1
                vectors[2, 1] = -1

        elif intercepts[2] != 0:
            if intercepts[2] > 0:
                vectors[0, 0] = 1
                vectors[1, 1] = 1
                vectors[2, 2] = 1
            else:
                vectors[0, 1] = 1
                vectors[1, 0] = 1
                vectors[2, 2] = -1

    elif count == 2:
        # Two-axis intercept
        if intercepts[2] == 0:
            s = signs[[0, 1]]
            vectors[0, 2] = s[0] * s[1]
            vectors[1, 0] = intercepts[0]
            vectors[1, 1] = -intercepts[1]
            vectors[2, 0] = s[0]
            vectors[2, 1] = s[1]

        elif intercepts[1] == 0:
            s = signs[[0, 2]]
            vectors[0, 1] = s[0] * s[1]
            vectors[1, 0] = -intercepts[0]
            vectors[1, 2] = intercepts[2]
            vectors[2, 0] = s[0]
            vectors[2, 2] = s[1]

        elif intercepts[0] == 0:
            s = signs[[1, 2]]
            vectors[0, 0] = s[0] * s[1]
            vectors[1, 1] = intercepts[1]
            vectors[1, 2] = -intercepts[2]
            vectors[2, 1] = s[0]
            vectors[2, 2] = s[1]

    else:
        # All three axes are intersected
        s = signs
        if s[2] > 0:
            vectors[0, 1] = s[0] * intercepts[1]
            vectors[0, 2] = -s[0] * intercepts[2]
            vectors[1, 0] = -s[1] * intercepts[0]
            vectors[1, 2] = s[1] * intercepts[2]
        else:
            vectors[0, 0] = -s[0] * intercepts[0]
            vectors[0, 2] = s[0] * intercepts[2]
            vectors[1, 1] = s[1] * intercepts[1]
            vectors[1, 2] = -s[1] * intercepts[2]

        vectors[2, :] = s

    return vectors


def get_boundary_box(int_vecs: np.ndarray) -> np.ndarray:
    """Calculate bounding box min/max per axis from 3 lattice vectors."""
    vecs_by_axis = int_vecs.T
    bound_box = np.zeros((3, 2), dtype=int)
    bound_box[:, 1] = np.sum(np.where(vecs_by_axis > 0, vecs_by_axis, 0), axis=1)
    bound_box[:, 0] = np.sum(np.where(vecs_by_axis < 0, vecs_by_axis, 0), axis=1)
    return bound_box


def get_lattice_vecs(
    atoms: Atoms,
    int_vecs: np.ndarray
) -> np.ndarray:
    """Calculate the new lattice vectors for the bulk cell."""
    latt_vecs_bulk = atoms.cell[:]
    lattice_vectors = np.dot(int_vecs, latt_vecs_bulk)
    lattice_vectors_oriented = orient_lattice_xy(lattice_vectors)

    return lattice_vectors_oriented


def convert_atoms(
    atoms: Atoms,
    int_vecs: np.ndarray,
    bound_box: np.ndarray,
    latt_vecs_new: np.ndarray,
    bond_threshold: float,
) -> SlabBulk:
    """Generate a translated cell with a new lattice vectors."""
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    inv_latt_vecs_old = np.linalg.inv(atoms.get_cell())
    inv_latt_int = np.linalg.inv(int_vecs)
    atoms_set = set()
    for ia in range(bound_box[0][0], bound_box[0][1] + 1):
        for ib in range(bound_box[1][0], bound_box[1][1] + 1):
            for ic in range(bound_box[2][0], bound_box[2][1] + 1):
                shift = np.array([ia, ib, ic])
                for symbol, position in zip(symbols, positions):
                    abc1 = np.dot(position, inv_latt_vecs_old)
                    abc2 = np.dot(abc1 + shift, inv_latt_int)
                    if np.all((-PACK_THR <= abc2) & (abc2 < 1.0 + PACK_THR)):
                        shifted_abc = abc2 - np.floor(abc2)
                        dz = (1.0 - shifted_abc[2]) * latt_vecs_new[2, 2]
                        if dz < POSIT_THR:
                            shifted_abc[2] -= 1.0
                        cartesian = np.dot(shifted_abc, latt_vecs_new)
                        atoms_set.add(SlabAtom(symbol, cartesian))

    return SlabBulk(latt_vecs_new, list(atoms_set), [], bond_threshold)


def convert_lattice_with_hkl_normal(
    atoms: Atoms, h: int, k: int, l: int, bond_threshold: float
) -> SlabBulk:
    """
    Convert the orientation of a crystal structure so that the (hkl) plane 
    becomes the top surface, i.e., the surface normal is aligned with [hkl].
    """
    if h == 0 and k == 0 and l == 0:
        raise ValueError("Miller indices [0, 0, 0] are not allowed.")
    positions: np.ndarray = atoms.get_positions()
    if positions.size == 0:
        raise ValueError("Given atoms object is blank.")
    
    basis_trans_matrix = get_basis_trans_matrix(get_intercepts(h, k, l))
    bound_box = get_boundary_box(basis_trans_matrix)
    latt_vecs_new = get_lattice_vecs(atoms, basis_trans_matrix)
    converted_cell = convert_atoms(
        atoms, basis_trans_matrix, bound_box, latt_vecs_new, bond_threshold
    )

    return converted_cell
