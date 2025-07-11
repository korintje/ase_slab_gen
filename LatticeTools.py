import numpy as np
from ase import Atoms as ASE_Atoms
from SlabModels import SlabAtom, SlabBulk
from math import gcd
from functools import reduce

BOHR_RADIUS_SI = 0.52917720859E-10 # m
BOHR_RADIUS_CM = BOHR_RADIUS_SI * 100.0
BOHR_RADIUS_ANGS = BOHR_RADIUS_CM * 1.0E8
PACK_THR: float = 1.0e-6 # internal coordinate
POSIT_THR = 0.01


def check_cell(lattice_vecs: np.ndarray) -> bool:
    """Check if lattice vectors array is valid."""
    if lattice_vecs.shape != (3, 3):
        return False
    return True


def get_lattice_consts(lattice_vecs: np.ndarray) -> np.ndarray:
    """Calculate lattice constants: (a, b, c, cos_alpha, cos_beta, cos_gamma)."""
    if not check_cell(lattice_vecs):
        return np.array([-1., -1., -1., 1., 1., 1.])
    a, b, c = np.linalg.norm(lattice_vecs, axis=1)
    cos_alpha = np.dot(lattice_vecs[1], lattice_vecs[2]) / (b * c) if b > 0 and c > 0 else 1.0
    cos_beta = np.dot(lattice_vecs[0], lattice_vecs[2]) / (a * c) if a > 0 and c > 0 else 1.0
    cos_gamma = np.dot(lattice_vecs[0], lattice_vecs[1]) / (a * b) if a > 0 and b > 0 else 1.0
    return np.array([a, b, c, cos_alpha, cos_beta, cos_gamma])


def get_cell_dm_14(lattice_vecs: np.ndarray) -> np.ndarray | None:
    """Convert lattice vectors to cell parameters in quantum espresso style."""
    if not check_cell(lattice_vecs):
        return None
    a, b, c, cos_alpha, cos_beta, cos_gamma = get_lattice_consts(lattice_vecs)
    celldm = np.array([
        a / BOHR_RADIUS_ANGS,
        b / a if a != 0 else 0.0,
        c / a if a != 0 else 0.0,
        cos_alpha,
        cos_beta,
        cos_gamma,
    ])
    return celldm


def get_cell_14(celldm: np.ndarray) -> np.ndarray | None:
    """Reconstruct lattice vectors."""
    if celldm is None or len(celldm) < 6 or celldm[0] == 0.0:
        return None
    a = celldm[0]
    b_a, c_a = celldm[1], celldm[2]
    cos_alpha, cos_beta, cos_gamma = celldm[3], celldm[4], celldm[5]

    # Validate cosine values
    if any(abs(x) > 1 for x in (cos_alpha, cos_beta, cos_gamma)):
        return None

    # Calculate volume factor terms safely
    term1 = np.sqrt(1.0 - cos_gamma ** 2)
    if term1 == 0:
        return None

    val = 1.0 + 2.0 * cos_alpha * cos_beta * cos_gamma - (cos_alpha ** 2 + cos_beta ** 2 + cos_gamma ** 2)
    if val < 0:
        return None
    term2 = np.sqrt(val) / term1

    lattice_vecs = np.zeros((3, 3), dtype=float)
    lattice_vecs[0] = [a, 0.0, 0.0]
    lattice_vecs[1] = [b_a * a * cos_gamma, b_a * a * term1, 0.0]
    lattice_vecs[2] = [c_a * a * cos_beta,
                       c_a * a * (cos_alpha - cos_beta * cos_gamma) / term1,
                       c_a * a * term2]
    return lattice_vecs * BOHR_RADIUS_ANGS


def lcm(a, b):
    """Compute least common multiple of two integers."""
    return abs(a * b) // gcd(a, b) if a and b else abs(a or b)


def lcm_multiple(numbers):
    """Compute the least common multiple (LCM) of a list of integers."""
    return reduce(lcm, numbers, 1)


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
        scale = lcm_multiple(nonzero)
        intercepts = np.array([
            scale // v if v != 0 else 0
            for v in miller
        ])

    return intercepts


def get_int_vecs(intercepts: np.ndarray) -> np.ndarray:
    """
    Construct a set of 3 integer vectors aligned with the Miller plane,
    based on the integer axis intercepts.
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
    """
    Calculate bounding box min/max per axis from 3 lattice vectors.
    Parameters:
        int_vecs (np.ndarray): 3 Integer vectors
    Returns:
        np.ndarray: shape (3, 2), min and max per axis
    """
    vecs_by_axis = int_vecs.T
    bound_box = np.zeros((3, 2), dtype=int)
    bound_box[:, 1] = np.sum(np.where(vecs_by_axis > 0, vecs_by_axis, 0), axis=1)
    bound_box[:, 0] = np.sum(np.where(vecs_by_axis < 0, vecs_by_axis, 0), axis=1)
    return bound_box


def get_lattice_vecs(
    atoms: ASE_Atoms,
    int_vecs: np.ndarray
) -> np.ndarray:
    """
    Calculate the new lattice vectors for the bulk cell after applying
    integer linear combinations of the original lattice vectors.
    Parameters:
        atoms (Atoms): ASE Atoms object containing the original cell.
        int_vecs (np.ndarray): Integer vectors.
    Returns:
        np.ndarray: New lattice vectors array with shape (3, 3).
    """
    latt_vecs_bulk = atoms.cell[:]

    # Calculate new lattice vectors by multiplying integer matrix by original vectors
    # (integer linear combinations of the original lattice vectors)
    latt_unit0 = np.dot(int_vecs, latt_vecs_bulk)

    # Extract lattice constants (lengths and angles) from the new lattice vectors
    latt_consts = get_cell_dm_14(latt_unit0)

    # Reconstruct lattice vectors from lattice constants to ensure consistent cell format
    latt_vecs = get_cell_14(latt_consts)

    return latt_vecs


def get_converted_atoms(
    atoms: ASE_Atoms,
    int_vecs: np.ndarray,
    bound_box: np.ndarray,
    latt_vecs_new: np.ndarray,
) -> SlabBulk:
    """
    Generate a transformed atomic slab by translating and filtering atoms
    into a new unit cell defined by transformed lattice vectors.
    Parameters:
        atoms (ASE_Atoms): Original ASE Atoms object.
        int_vecs (np.ndarray): Integer vectors used for the lattice transformation matrix.
        bound_box (np.ndarray): 3x2 array of integer boundaries for search range.
        latt_vecs_new (np.ndarray): New lattice vectors of the slab cell.
    Returns:
        SlabBulk: New atomic slab object with transformed atoms and lattice.
    """

    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    
    # Inverse of original cell vectors (Cartesian → fractional)
    inv_latt_vecs_old = np.linalg.inv(atoms.get_cell())
    
    # Inverse of integer transformation matrix
    inv_latt_int = np.linalg.inv(int_vecs)

    atoms_set = set()

    for ia in range(bound_box[0][0], bound_box[0][1] + 1):
        for ib in range(bound_box[1][0], bound_box[1][1] + 1):
            for ic in range(bound_box[2][0], bound_box[2][1] + 1):
                shift = np.array([ia, ib, ic])
                for symbol, position in zip(symbols, positions):
                    # Convert to fractional coordinates of original cell
                    abc1 = np.dot(position, inv_latt_vecs_old)
                    
                    # Apply integer translation and convert to new fractional coordinates
                    abc2 = np.dot(abc1 + shift, inv_latt_int)

                    if np.all((-PACK_THR <= abc2) & (abc2 < 1.0 + PACK_THR)):
                        shifted_abc = abc2 - np.floor(abc2)

                        # z correction to keep within positive region
                        dz = (1.0 - shifted_abc[2]) * latt_vecs_new[2, 2]
                        if dz < POSIT_THR:
                            shifted_abc[2] -= 1.0

                        cartesian = np.dot(shifted_abc, latt_vecs_new)
                        atoms_set.add(SlabAtom(symbol, cartesian))

    return SlabBulk(latt_vecs_new, list(atoms_set), [])


def convert_lattice_with_hkl_normal(
    atoms: ASE_Atoms, h: int, k: int, l: int
) -> SlabBulk:
    """
    Convert the orientation of a crystal structure so that the (hkl) plane 
    becomes the top surface, i.e., the surface normal is aligned with [hkl].
    Parameters:
        atoms (Atoms): ASE Atoms object representing the bulk crystal.
        h (int): Miller index h.
        k (int): Miller index k.
        l (int): Miller index l.
    Returns:
        SlabBulk: New Atoms object with the cell reoriented so that the 
        (hkl) plane is aligned as the surface.
    """

    # Validate Miller indices
    if h == 0 and k == 0 and l == 0:
        raise ValueError("Miller indices [0, 0, 0] are not allowed.")

    # Validate that the atoms object is not empty
    positions: np.ndarray = atoms.get_positions()
    if positions.size == 0:
        raise ValueError("Given atoms object is blank.")
    
    # Generate integer vectors used for the lattice transformation
    int_vecs = get_int_vecs(get_intercepts(h, k, l))

    # Calculate the bounding box for slicing the cell
    bound_box = get_boundary_box(int_vecs)

    # Determine new lattice vectors for the reoriented cell
    latt_vecs_new = get_lattice_vecs(atoms, int_vecs)

    # Apply the transformation and extract the rotated slab
    converted_cell = get_converted_atoms(
        atoms, int_vecs, bound_box, latt_vecs_new
    )

    return converted_cell
