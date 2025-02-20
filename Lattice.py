import numpy as np

BOHR_RADIUS_SI = 0.52917720859E-10 # m
BOHR_RADIUS_CM = BOHR_RADIUS_SI * 100.0
BOHR_RADIUS_ANGS = BOHR_RADIUS_CM * 1.0E8


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
