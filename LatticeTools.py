import numpy as np
from ase import Atoms as ASE_Atoms
from ase import Atom as ASE_Atom
from SlabModels import SlabAtom, SlabBulk

BOHR_RADIUS_SI = 0.52917720859E-10 # m
BOHR_RADIUS_CM = BOHR_RADIUS_SI * 100.0
BOHR_RADIUS_ANGS = BOHR_RADIUS_CM * 1.0E8
PACK_THR: float = 1.0e-6 # internal coordinate
POSIT_THR = 0.01


def check_cell(lattice_vecs: np.ndarray):
    
    if lattice_vecs.size == 0 or len(lattice_vecs) < 3:
        return False
    for lattice_vec in lattice_vecs:
        if lattice_vec.size == 0 or len(lattice_vec) < 3:
            return False
    return True


def get_lattice_consts(lattice_vecs: np.ndarray):

    if not check_cell(lattice_vecs):
        a, b, c, cos_alpha, cos_beta, cos_gamma = -1.0, -1.0, -1.0, 1.0, 1.0, 1.0
    else:    
        a = np.linalg.norm(lattice_vecs[0])
        b = np.linalg.norm(lattice_vecs[1])
        c = np.linalg.norm(lattice_vecs[2])
        
        if b <= 0.0 or c <= 0.0:
            cos_alpha = 1.0
        else:
            cos_alpha = np.dot(lattice_vecs[1], lattice_vecs[2]) / (b * c)

        if c <= 0.0 or a <= 0.0:
            cos_beta = 1.0
        else:
            cos_beta = np.dot(lattice_vecs[0], lattice_vecs[2]) / (a * c)

        if a <= 0.0 or b <= 0.0:
            cos_gamma = 1.0
        else:
            cos_gamma = np.dot(lattice_vecs[0], lattice_vecs[1]) / (a * b)
    
    return [a, b, c, cos_alpha, cos_beta, cos_gamma]


def get_cell_dm_14(lattice_vecs: np.ndarray):

    if not check_cell(lattice_vecs):
        return None

    a, b, c, cos_alpha, cos_beta, cos_gamma = get_lattice_consts(lattice_vecs)

    celldm = [
        a / BOHR_RADIUS_ANGS,
        b / a,
        c / a,
        cos_alpha,
        cos_beta,
        cos_gamma
    ]

    return celldm


def get_cell_14(celldm):

    if celldm is None or len(celldm) < 6:
        return None

    if celldm[0] == 0.0:
        return None

    lattice_vecs: np.ndarray = np.zeros((3, 3))

    term1 = 0.0
    term2 = 0.0

    if celldm[1] <= 0.0 or celldm[2] <= 0.0 or abs(celldm[3]) >= 1.0 or abs(celldm[4]) >= 1.0 or abs(celldm[5]) >= 1.0:
        return None
    term1 = np.sqrt(1.0 - celldm[5]**2)
    if term1 == 0.0:
        return None
    term2 = (1.0 + 2.0 * celldm[3] * celldm[4] * celldm[5]) - (celldm[3]**2 + celldm[4]**2 + celldm[5]**2)
    if term2 < 0.0:
        return None
    term2 = np.sqrt(term2 / term1**2)
    
    lattice_vecs[0][0] = celldm[0]
    lattice_vecs[1][0] = celldm[0] * celldm[1] * celldm[5]
    lattice_vecs[1][1] = celldm[0] * celldm[1] * term1
    lattice_vecs[2][0] = celldm[0] * celldm[2] * celldm[4]
    lattice_vecs[2][1] = celldm[0] * celldm[2] * (celldm[3] - celldm[4] * celldm[5]) / term1
    lattice_vecs[2][2] = celldm[0] * celldm[2] * term2

    lattice_vecs = lattice_vecs * BOHR_RADIUS_ANGS

    return lattice_vecs


def get_intercepts(h: int, k: int, l: int):
    """
    Calculate integer intercepts for the (hkl) Miller plane in a crystal lattice.
    This function determines the smallest integer scale such that the intercepts
    along x, y, and z axes become integers.

    Parameters:
        h (int): Miller index h
        k (int): Miller index k
        l (int): Miller index l

    Returns:
        Tuple containing:
            - num_intercept (int): Number of non-zero Miller indices
            - has_intercept1 (bool): True if h ≠ 0 (x-axis intercept exists)
            - has_intercept2 (bool): True if k ≠ 0 (y-axis intercept exists)
            - has_intercept3 (bool): True if l ≠ 0 (z-axis intercept exists)
            - intercept1 (int): Integer intercept along x-axis
            - intercept2 (int): Integer intercept along y-axis
            - intercept3 (int): Integer intercept along z-axis
    """

    # Initialize scale range for computing least common multiple (LCM)
    scale_min = 1
    scale_max = 1
    num_intercept = 0

    # Check for non-zero h (x-direction intercept)
    if h != 0:
        scale_min = max(scale_min, abs(h))
        scale_max *= abs(h)
        num_intercept += 1
        has_intercept1 = True
    else:
        has_intercept1 = False

    # Check for non-zero k (y-direction intercept)
    if k != 0:
        scale_min = max(scale_min, abs(k))
        scale_max *= abs(k)
        num_intercept += 1
        has_intercept2 = True
    else:
        has_intercept2 = False

    # Check for non-zero l (z-direction intercept)
    if l != 0:
        scale_min = max(scale_min, abs(l))
        scale_max *= abs(l)
        num_intercept += 1
        has_intercept3 = True
    else:
        has_intercept3 = False

    # Search for the smallest integer scale divisible by all non-zero indices
    # Equivalent to computing the least common multiple (LCM) of non-zero h, k, l
    scale = 0
    for i in range(scale_min, scale_max + 1):
        if has_intercept1 and (i % h) != 0:
            continue
        if has_intercept2 and (i % k) != 0:
            continue
        if has_intercept3 and (i % l) != 0:
            continue
        scale = i
        break

    # Compute integer intercepts along each axis based on the scale
    intercept1 = scale // h if has_intercept1 else 0
    intercept2 = scale // k if has_intercept2 else 0
    intercept3 = scale // l if has_intercept3 else 0

    return (
        num_intercept,
        has_intercept1,
        has_intercept2,
        has_intercept3,
        intercept1,
        intercept2,
        intercept3
    )


def get_vectors(
    num_intercept: int,
    has_intercept1: bool,
    has_intercept2: bool,
    has_intercept3: bool,
    intercept1: int,
    intercept2: int,
    intercept3: int
):
    """
    Generate three lattice vectors (vector1, vector2, vector3) that span the 
    transformed unit cell where the (hkl) plane is defined by the provided intercepts.

    Parameters:
        num_intercept (int): Number of non-zero Miller indices (i.e., plane intercepts).
        has_intercept1 (bool): True if intercept along x-axis exists (h ≠ 0).
        has_intercept2 (bool): True if intercept along y-axis exists (k ≠ 0).
        has_intercept3 (bool): True if intercept along z-axis exists (l ≠ 0).
        intercept1 (int): Integer intercept along x-axis.
        intercept2 (int): Integer intercept along y-axis.
        intercept3 (int): Integer intercept along z-axis.

    Returns:
        Tuple of np.ndarray:
            vector1 (np.ndarray): First in-plane vector (along the surface).
            vector2 (np.ndarray): Second in-plane vector (along the surface).
            vector3 (np.ndarray): Out-of-plane vector (normal to surface).
    """

    # Initialize the three vectors as integer arrays of shape (3,)
    vector1 = np.zeros(3, dtype=int)
    vector2 = np.zeros(3, dtype=int)
    vector3 = np.zeros(3, dtype=int)

    # Use specialized logic depending on the number of non-zero intercepts
    if num_intercept <= 1:
        # Special case: plane intersects only one axis
        # e.g., (1, 0, 0) — simple setup for 1D edge/face plane
        setup_vectors1(
            has_intercept1, has_intercept2, has_intercept3,
            intercept1, intercept2, intercept3,
            vector1, vector2, vector3
        )

    elif num_intercept <= 2:
        # Intermediate case: plane intersects two axes (edge cuts)
        # e.g., (1, 1, 0) — diagonal within a face
        setup_vectors2(
            has_intercept1, has_intercept2, has_intercept3,
            intercept1, intercept2, intercept3,
            vector1, vector2, vector3
        )

    else:
        # General case: plane intersects all three axes
        # e.g., (1, 1, 1) — full 3D corner cut
        setup_vectors3(
            intercept1, intercept2, intercept3,
            vector1, vector2, vector3
        )

    return vector1, vector2, vector3


def setup_vectors1(
    has_intercept1: bool,
    has_intercept2: bool,
    has_intercept3: bool,
    intercept1: int,
    intercept2: int,
    intercept3: int,
    vector1: np.ndarray,
    vector2: np.ndarray,
    vector3: np.ndarray
):
    """
    Define lattice vectors when the Miller plane intersects only one axis.
    This is a special case for (hkl) planes like (1, 0, 0), (0, 1, 0), (0, 0, 1).

    Parameters:
        has_intercept1 (bool): True if the plane intersects the x-axis (h ≠ 0)
        has_intercept2 (bool): True if the plane intersects the y-axis (k ≠ 0)
        has_intercept3 (bool): True if the plane intersects the z-axis (l ≠ 0)
        intercept1 (int): Integer intercept along the x-axis
        intercept2 (int): Integer intercept along the y-axis
        intercept3 (int): Integer intercept along the z-axis
        vector1 (np.ndarray): Output vector (in-plane direction)
        vector2 (np.ndarray): Output vector (in-plane direction)
        vector3 (np.ndarray): Output vector (normal to surface)
    """

    # Case 1: Plane intersects x-axis only (e.g., (h, 0, 0))
    if has_intercept1:
        if intercept1 > 0:
            # Positive x-intercept → surface normal points in +x
            vector1[1] = 1  # y-direction
            vector2[2] = 1  # z-direction
            vector3[0] = 1  # x-direction (normal)
        else:
            # Negative x-intercept → flip x-normal
            vector1[2] = 1  # z-direction
            vector2[1] = 1  # y-direction
            vector3[0] = -1  # -x-direction (normal)

    # Case 2: Plane intersects y-axis only (e.g., (0, k, 0))
    elif has_intercept2:
        if intercept2 > 0:
            # Positive y-intercept → normal in +y
            vector1[2] = 1  # z-direction
            vector2[0] = 1  # x-direction
            vector3[1] = 1  # y-direction (normal)
        else:
            # Negative y-intercept → flip y-normal
            vector1[0] = 1  # x-direction
            vector2[2] = 1  # z-direction
            vector3[1] = -1  # -y-direction (normal)

    # Case 3: Plane intersects z-axis only (e.g., (0, 0, l))
    elif has_intercept3:
        if intercept3 > 0:
            # Positive z-intercept → normal in +z
            vector1[0] = 1  # x-direction
            vector2[1] = 1  # y-direction
            vector3[2] = 1  # z-direction (normal)
        else:
            # Negative z-intercept → flip z-normal
            vector1[1] = 1  # y-direction
            vector2[0] = 1  # x-direction
            vector3[2] = -1  # -z-direction (normal)


def setup_vectors2(
    has_intercept1: bool,
    has_intercept2: bool,
    has_intercept3: bool,
    intercept1: int,
    intercept2: int,
    intercept3: int,
    vector1: np.ndarray,
    vector2: np.ndarray,
    vector3: np.ndarray
):
    """
    Define lattice vectors when the Miller plane intersects exactly two axes.
    This includes planes like (1,1,0), (1,0,1), or (0,1,1).

    Parameters:
        has_intercept1 (bool): True if h ≠ 0 (x-axis intercept exists)
        has_intercept2 (bool): True if k ≠ 0 (y-axis intercept exists)
        has_intercept3 (bool): True if l ≠ 0 (z-axis intercept exists)
        intercept1 (int): Integer intercept along x-axis
        intercept2 (int): Integer intercept along y-axis
        intercept3 (int): Integer intercept along z-axis
        vector1 (np.ndarray): Output vector (in-plane direction)
        vector2 (np.ndarray): Output vector (in-plane direction)
        vector3 (np.ndarray): Output vector (normal to surface)
    """

    # Case: Plane lies in x-y plane (intercepts x and y only)
    if not has_intercept3:
        sign1 = int((intercept1 > 0) - (intercept1 < 0))
        sign2 = int((intercept2 > 0) - (intercept2 < 0))

        # In-plane vector perpendicular to (h,k,0)
        vector1[2] = sign1 * sign2  # z direction (out of plane)
        vector2[0] = intercept1     # x component of second in-plane vector
        vector2[1] = -intercept2    # y component with opposite sign
        vector3[0] = sign1          # x component of surface normal
        vector3[1] = sign2          # y component of surface normal

    # Case: Plane lies in x-z plane (intercepts x and z only)
    elif not has_intercept2:
        sign1 = int((intercept1 > 0) - (intercept1 < 0))
        sign3 = int((intercept3 > 0) - (intercept3 < 0))

        # In-plane vector perpendicular to (h,0,l)
        vector1[1] = sign1 * sign3  # y direction (out of plane)
        vector2[0] = -intercept1    # x component
        vector2[2] = intercept3     # z component
        vector3[0] = sign1          # x component of surface normal
        vector3[2] = sign3          # z component of surface normal

    # Case: Plane lies in y-z plane (intercepts y and z only)
    elif not has_intercept1:
        sign2 = int((intercept2 > 0) - (intercept2 < 0))
        sign3 = int((intercept3 > 0) - (intercept3 < 0))

        # In-plane vector perpendicular to (0,k,l)
        vector1[0] = sign2 * sign3  # x direction (out of plane)
        vector2[1] = intercept2     # y component
        vector2[2] = -intercept3    # z component
        vector3[1] = sign2          # y component of surface normal
        vector3[2] = sign3          # z component of surface normal


def setup_vectors3(
    intercept1: int,
    intercept2: int,
    intercept3: int,
    vector1: np.ndarray,
    vector2: np.ndarray,
    vector3: np.ndarray,
):
    """
    Define lattice vectors when the Miller plane intersects all three axes.
    This is the general case for (hkl) planes such as (1,1,1).

    Parameters:
        intercept1 (int): Integer intercept along the x-axis
        intercept2 (int): Integer intercept along the y-axis
        intercept3 (int): Integer intercept along the z-axis
        vector1 (np.ndarray): Output vector (in-plane direction)
        vector2 (np.ndarray): Output vector (in-plane direction)
        vector3 (np.ndarray): Output vector (normal to surface)
    """

    # Determine the signs of each intercept (+1, -1, or 0)
    sign1 = (intercept1 > 0) - (intercept1 < 0)
    sign2 = (intercept2 > 0) - (intercept2 < 0)
    sign3 = (intercept3 > 0) - (intercept3 < 0)

    # Choose in-plane vectors based on the direction of the z-axis intercept
    if sign3 > 0:
        # If z-intercept is positive, orient vectors to preserve right-handed system
        vector1[1] = sign1 * intercept2   # y-component of vector1
        vector1[2] = -sign1 * intercept3  # z-component of vector1
        vector2[0] = -sign2 * intercept1  # x-component of vector2
        vector2[2] = sign2 * intercept3   # z-component of vector2
    else:
        # If z-intercept is zero or negative, use a different basis to avoid degeneracy
        vector1[0] = -sign1 * intercept1  # x-component of vector1
        vector1[2] = sign1 * intercept3   # z-component of vector1
        vector2[1] = sign2 * intercept2   # y-component of vector2
        vector2[2] = -sign2 * intercept3  # z-component of vector2

    # Out-of-plane vector: normal to the (hkl) plane
    vector3[0] = sign1  # x-direction
    vector3[1] = sign2  # y-direction
    vector3[2] = sign3  # z-direction


def get_boundary_box(
    vector1: np.ndarray,
    vector2: np.ndarray,
    vector3: np.ndarray
) -> np.ndarray:
    """
    Calculate the bounding box limits along each Cartesian axis based on
    the three lattice vectors defining a unit cell.

    The bounding box is represented as a (3, 2) array, where each row
    corresponds to an axis (x, y, z), and the two columns represent the
    minimum and maximum coordinate extents respectively.

    Parameters:
        vector1 (np.ndarray): First lattice vector (length 3)
        vector2 (np.ndarray): Second lattice vector (length 3)
        vector3 (np.ndarray): Third lattice vector (length 3)

    Returns:
        np.ndarray: Bounding box array of shape (3, 2),
                    with min and max values along each axis.
    """

    # Initialize bounding box array with zeros:
    # Each row for x,y,z axes; columns: [min, max]
    bound_box = np.zeros((3, 2), dtype=int)

    # Iterate over x,y,z components (indices 0,1,2)
    for i in range(3):
        # For each vector, add negative components to min bound,
        # positive components to max bound.

        # vector1 component
        if vector1[i] < 0:
            bound_box[i][0] += vector1[i]
        else:
            bound_box[i][1] += vector1[i]

        # vector2 component
        if vector2[i] < 0:
            bound_box[i][0] += vector2[i]
        else:
            bound_box[i][1] += vector2[i]

        # vector3 component
        if vector3[i] < 0:
            bound_box[i][0] += vector3[i]
        else:
            bound_box[i][1] += vector3[i]

    return bound_box


def get_lattice_vecs(
    atoms: ASE_Atoms,
    vector1: np.ndarray,
    vector2: np.ndarray,
    vector3: np.ndarray
) -> np.ndarray:
    """
    Calculate the new lattice vectors for the bulk cell after applying
    integer linear combinations of the original lattice vectors.

    Parameters:
        atoms (Atoms): ASE Atoms object containing the original cell.
        vector1 (np.ndarray): First integer vector (length 3).
        vector2 (np.ndarray): Second integer vector (length 3).
        vector3 (np.ndarray): Third integer vector (length 3).

    Returns:
        np.ndarray: New lattice vectors array with shape (3, 3).
    """

    # Extract the original lattice vectors from the Atoms object
    latt_vecs_bulk = atoms.cell[:]

    # Calculate new lattice vectors by multiplying integer matrix by original vectors
    # (integer linear combinations of the original lattice vectors)
    latt_int = np.stack([vector1, vector2, vector3])
    latt_unit0 = np.dot(latt_int, latt_vecs_bulk)

    # Extract lattice constants (lengths and angles) from the new lattice vectors
    latt_consts = get_cell_dm_14(latt_unit0)

    # Reconstruct lattice vectors from lattice constants to ensure consistent cell format
    latt_vecs = get_cell_14(latt_consts)

    return latt_vecs


def get_converted_atoms(
    atoms: ASE_Atoms,
    vector1: np.ndarray,
    vector2: np.ndarray,
    vector3: np.ndarray,
    bound_box: np.ndarray,
    latt_vecs_new: np.ndarray,
) -> SlabBulk:

    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    inv_latt_vecs_old = np.linalg.inv(atoms.get_cell())
    inv_latt_int = np.linalg.inv(
        np.stack([vector1, vector2, vector3])
    )

    atoms_set = set()

    add_count = 0
    
    for ia in range(bound_box[0][0], bound_box[0][1] + 1):
        for ib in range(bound_box[1][0], bound_box[1][1] + 1):
            for ic in range(bound_box[2][0], bound_box[2][1] + 1):
                for symbol, position in zip(symbols, positions):
                    abc1 = np.dot(position, inv_latt_vecs_old)
                    abc2 = np.dot(abc1 + np.array([ia, ib, ic]), inv_latt_int)
                    if (-PACK_THR <= abc2[0] < 1.0 + PACK_THR and 
                        -PACK_THR <= abc2[1] < 1.0 + PACK_THR and 
                        -PACK_THR <= abc2[2] < 1.0 + PACK_THR):
                        shifted_abc = abc2 - np.floor(abc2)
                        dc = 1.0 - shifted_abc[2]
                        dz = dc * latt_vecs_new[2][2]
                        if dz < POSIT_THR:
                            shifted_abc[2] -= 1.0
                        atom = SlabAtom(symbol, np.dot(shifted_abc, latt_vecs_new))
                        add_count += 1
                        atoms_set.add(atom)
    
    converted_atoms = SlabBulk(latt_vecs_new, list(atoms_set), [])

    return converted_atoms


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

    # Calculate intercepts for the (hkl) plane and
    # Generate new basis vectors from the (hkl) intercepts
    vector1, vector2, vector3 = get_vectors(
        *get_intercepts(h, k, l)
    )

    # Calculate the bounding box for slicing the cell
    bound_box = get_boundary_box(vector1, vector2, vector3)

    # Determine new lattice vectors for the reoriented cell
    latt_vecs_new = get_lattice_vecs(
        atoms, vector1, vector2, vector3
    )

    # Apply the transformation and extract the rotated slab
    converted_cell = get_converted_atoms(
        atoms, vector1, vector2, vector3, bound_box, latt_vecs_new
    )

    return converted_cell