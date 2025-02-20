import numpy as np
from ASE import Atoms as ASE_Atoms
from ASE import Atom as ASE_Atom


BOHR_RADIUS_SI = 0.52917720859E-10 # m
BOHR_RADIUS_CM = BOHR_RADIUS_SI * 100.0
BOHR_RADIUS_ANGS = BOHR_RADIUS_CM * 1.0E8
PACK_THR: float = 1.0e-6     # internal coordinate
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


def convert_by_hkl(cell: ASE_Atoms, h: int, k: int, l: int):

    if h == 0 and k == 0 and l == 0:
        raise ValueError("Miller indices [0, 0, 0] is not allowed.")
    
    positions: np.ndarray = cell.get_positions()
    if positions.size == 0:
        raise ValueError("Given atoms object is blank.")
    
    num_intercept, has_intercept1, has_intercept2, has_intercept3, intercept1, intercept2, intercept3 = get_intercepts(h, k, l)
    vector1, vector2, vector3 = get_vectors(num_intercept, has_intercept1, has_intercept2, has_intercept3, intercept1, intercept2, intercept3)
    bound_box = get_boundary_box(vector1, vector2, vector3)
    latt_unit = setup_lattice(cell, vector1, vector2, vector3)
    converted_cell = setup_unit_atoms_in_slab(cell, vector1, vector2, vector3, bound_box, latt_unit)

    return converted_cell


def get_intercepts(miller1, miller2, miller3):
    scale_min = 1
    scale_max = 1
    num_intercept = 0

    if miller1 != 0:
        scale_min = max(scale_min, abs(miller1))
        scale_max *= abs(miller1)
        num_intercept += 1
        has_intercept1 = True
    else:
        has_intercept1 = False

    if miller2 != 0:
        scale_min = max(scale_min, abs(miller2))
        scale_max *= abs(miller2)
        num_intercept += 1
        has_intercept2 = True
    else:
        has_intercept2 = False

    if miller3 != 0:
        scale_min = max(scale_min, abs(miller3))
        scale_max *= abs(miller3)
        num_intercept += 1
        has_intercept3 = True
    else:
        has_intercept3 = False

    if scale_min < 1:
        raise ValueError("scaleMin is not positive.")

    if scale_max < scale_min:
        raise ValueError("scaleMax < scaleMin.")

    if num_intercept < 1:
        raise ValueError("there are no intercepts.")

    scale = 0
    for i in range(scale_min, scale_max + 1):
        if has_intercept1 and (i % miller1) != 0:
            continue
        if has_intercept2 and (i % miller2) != 0:
            continue
        if has_intercept3 and (i % miller3) != 0:
            continue

        scale = i
        break

    if scale < 1:
        raise ValueError("cannot detect scale.")

    intercept1 = scale // miller1 if has_intercept1 else 0
    intercept2 = scale // miller2 if has_intercept2 else 0
    intercept3 = scale // miller3 if has_intercept3 else 0

    return num_intercept, has_intercept1, has_intercept2, has_intercept3, intercept1, intercept2, intercept3


def get_vectors(num_intercept, has_intercept1, has_intercept2, has_intercept3, intercept1, intercept2, intercept3):

    vector1: np.ndarray = np.zeros(3, dtype = int)
    vector2: np.ndarray = np.zeros(3, dtype = int)
    vector3: np.ndarray = np.zeros(3, dtype = int)

    if num_intercept <= 1:
        setup_vectors1(has_intercept1, has_intercept2, has_intercept3, intercept1, intercept2, intercept3, vector1, vector2, vector3)
    elif num_intercept <= 2:
        setup_vectors2(has_intercept1, has_intercept2, has_intercept3, intercept1, intercept2, intercept3, vector1, vector2, vector3)
    else:
        setup_vectors3(intercept1, intercept2, intercept3, vector1, vector2, vector3)
    
    return vector1, vector2, vector3


def setup_vectors1(has_intercept1, has_intercept2, has_intercept3, intercept1, intercept2, intercept3, vector1, vector2, vector3):
    if has_intercept1:
        if intercept1 > 0:
            vector1[1] = 1
            vector2[2] = 1
            vector3[0] = 1
        else:
            vector1[2] = 1
            vector2[1] = 1
            vector3[0] = -1

    elif has_intercept2:
        if intercept2 > 0:
            vector1[2] = 1
            vector2[0] = 1
            vector3[1] = 1
        else:
            vector1[0] = 1
            vector2[2] = 1
            vector3[1] = -1

    elif has_intercept3:
        if intercept3 > 0:
            vector1[0] = 1
            vector2[1] = 1
            vector3[2] = 1
        else:
            vector1[1] = 1
            vector2[0] = 1
            vector3[2] = -1


def setup_vectors2(has_intercept1, has_intercept2, has_intercept3, intercept1, intercept2, intercept3, vector1, vector2, vector3):
    if not has_intercept3:  # cat in A-B plane
        sign1 = int((intercept1 > 0) - (intercept1 < 0))
        sign2 = int((intercept2 > 0) - (intercept2 < 0))
        vector1[2] = sign1 * sign2
        vector2[0] = intercept1
        vector2[1] = -intercept2
        vector3[0] = sign1
        vector3[1] = sign2

    elif not has_intercept2:  # cat in A-C plane
        sign1 = int((intercept1 > 0) - (intercept1 < 0))
        sign3 = int((intercept3 > 0) - (intercept3 < 0))
        vector1[1] = sign1 * sign3
        vector2[0] = -intercept1
        vector2[2] = intercept3
        vector3[0] = sign1
        vector3[2] = sign3

    elif not has_intercept1:  # cat in B-C plane
        sign2 = int((intercept2 > 0) - (intercept2 < 0))
        sign3 = int((intercept3 > 0) - (intercept3 < 0))
        vector1[0] = sign2 * sign3
        vector2[1] = intercept2
        vector2[2] = -intercept3
        vector3[1] = sign2
        vector3[2] = sign3


def setup_vectors3(intercept1, intercept2, intercept3, vector1, vector2, vector3):
    sign1 = (intercept1 > 0) - (intercept1 < 0)
    sign2 = (intercept2 > 0) - (intercept2 < 0)
    sign3 = (intercept3 > 0) - (intercept3 < 0)

    if sign3 > 0:
        vector1[1] = sign1 * intercept2
        vector1[2] = -sign1 * intercept3
        vector2[0] = -sign2 * intercept1
        vector2[2] = sign2 * intercept3
    else:
        vector1[0] = -sign1 * intercept1
        vector1[2] = sign1 * intercept3
        vector2[1] = sign2 * intercept2
        vector2[2] = -sign2 * intercept3

    vector3[0] = sign1
    vector3[1] = sign2
    vector3[2] = sign3


def get_boundary_box(vector1, vector2, vector3):

    bound_box = [[0, 0] for _ in range(3)]

    for i in range(3):
        if vector1[i] < 0:
            bound_box[i][0] += vector1[i]
        else:
            bound_box[i][1] += vector1[i]

        if vector2[i] < 0:
            bound_box[i][0] += vector2[i]
        else:
            bound_box[i][1] += vector2[i]

        if vector3[i] < 0:
            bound_box[i][0] += vector3[i]
        else:
            bound_box[i][1] += vector3[i]
    
    return bound_box


def setup_lattice(cell: ASE_Atoms, vector1, vector2, vector3):

    latt_int = np.stack([vector1, vector2, vector3])
    lattice: np.ndarray = cell.cell[:]
    latt_unit0 = np.dot(latt_int, lattice)
    latt_const = None if latt_unit0.size == 0 else get_cell_dm_14(latt_unit0)
    if latt_const is None or len(latt_const) < 6:
        raise ValueError("Lattice constants are invalid.")

    latt_unit = get_cell_14(latt_const)
    if latt_unit is None or latt_unit.size == 0 or len(latt_unit) < 3:
        raise ValueError("Lattice vectors are invalid.")
    for i in range(3):
        if latt_unit[i] is None or len(latt_unit[i]) < 3:
            raise ValueError(f"Lattice vector {i} is invalid.")
    
    return latt_unit


def setup_unit_atoms_in_slab(cell, vector1, vector2, vector3, bound_box, latt_unit):

    positions: np.ndarray = cell.get_positions()
    symbols: np.ndarray = cell.get_chemical_symbols()
    lattice_vecs: np.ndarray = cell.get_cell()
    inv_lattice_vecs = np.linalg.inv(lattice_vecs)

    latt_int = np.stack([vector1, vector2, vector3])
    inv_latt = np.linalg.inv(latt_int)

    unit_atoms_set = set()
    for ia in range(bound_box[0][0], bound_box[0][1] + 1):
        for ib in range(bound_box[1][0], bound_box[1][1] + 1):
            for ic in range(bound_box[2][0], bound_box[2][1] + 1):
                for symbol, position in zip(symbols, positions):
                    abc1 = np.dot(position, inv_lattice_vecs)
                    abc2 = np.dot(abc1 + np.array([ia, ib, ic]), inv_latt)
                    if (-PACK_THR <= abc2[0] < 1.0 + PACK_THR and 
                        -PACK_THR <= abc2[1] < 1.0 + PACK_THR and 
                        -PACK_THR <= abc2[2] < 1.0 + PACK_THR):
                        shifted_abc = abc2 - np.floor(abc2)
                        dc = 1.0 - shifted_abc[2]
                        dz = dc * latt_unit[2][2]
                        if dz < POSIT_THR:
                            shifted_abc[2] -= 1.0
                        atom = ASE_Atom(symbol, np.dot(shifted_abc, latt_unit))
                        unit_atoms_set.add(atom)
    
    slab_atoms = ASE_Atoms()
    slab_atoms.set_cell(latt_unit)
    for atom in unit_atoms_set:
        slab_atoms.append(atom)

    return slab_atoms
