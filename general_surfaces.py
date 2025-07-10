import numpy as np
from ase.atoms import Atoms
from SlabGenom import SlabGenom
from SlabModels import SlabAtom, SlabBulk, Slab


BOHR_RADIUS_SI = 0.52917720859E-10 # m
BOHR_RADIUS_CM = BOHR_RADIUS_SI * 100.0
BOHR_RADIUS_ANGS = BOHR_RADIUS_CM * 1.0E8
PACK_THR: float = 1.0e-6     # internal coordinate
POSIT_THR = 0.01
STEP_FOR_GENOMS: float = 0.50  # angstromset()


def surfaces(lattice, indices, layers: int, vacuum: float=0.0, adsorbates=None) -> Atoms:
    """Create surfaces from a given lattice and Miller indices.

    lattice: Atoms object or str
        Bulk lattice structure of alloy or pure metal.  Note that the
        unit-cell must be the conventional cell - not the primitive cell.
        One can also give the chemical symbol as a string, in which case the
        correct bulk lattice will be generated automatically.
    indices: sequence of three int
        Surface normal in Miller indices (h,k,l).
    layers: int
        Number of equivalent layers of the slab.
    vacuum: float
        Amount of vacuum added on both sides of the slab.
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
    hkl_directed_lattice = convert_lattice_with_hkl_normal(atoms, *indices)
    slabs = get_slabs(hkl_directed_lattice, layers)
    surfaces = [slab.to_atoms() for slab in slabs]
    
    return surfaces


# Retrun crystal lattice whose upper plane is perpendicular to the normal vector specified by h, k, l
def convert_lattice_with_hkl_normal(atoms: Atoms, h: int, k: int, l: int) -> SlabBulk:

    if h == 0 and k == 0 and l == 0:
        raise ValueError("Miller indices [0, 0, 0] is not allowed.")
    
    positions: np.ndarray = atoms.get_positions()
    if positions.size == 0:
        raise ValueError("Given atoms object is blank.")
    
    num_intercept, has_intercept1, has_intercept2, has_intercept3, intercept1, intercept2, intercept3 = get_intercepts(h, k, l)
    vector1, vector2, vector3 = get_vectors(num_intercept, has_intercept1, has_intercept2, has_intercept3, intercept1, intercept2, intercept3)
    bound_box = get_boundary_box(vector1, vector2, vector3)
    latt_vecs_new = get_lattice_vecs(atoms, vector1, vector2, vector3)
    converted_cell = get_converted_atoms(atoms, vector1, vector2, vector3, bound_box, latt_vecs_new)

    return converted_cell


def get_intercepts(h: int, k: int, l: int):
    scale_min = 1
    scale_max = 1
    num_intercept = 0

    if h != 0:
        scale_min = max(scale_min, abs(h))
        scale_max *= abs(h)
        num_intercept += 1
        has_intercept1 = True
    else:
        has_intercept1 = False

    if k != 0:
        scale_min = max(scale_min, abs(k))
        scale_max *= abs(k)
        num_intercept += 1
        has_intercept2 = True
    else:
        has_intercept2 = False

    if l != 0:
        scale_min = max(scale_min, abs(l))
        scale_max *= abs(l)
        num_intercept += 1
        has_intercept3 = True
    else:
        has_intercept3 = False

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

    intercept1 = scale // h if has_intercept1 else 0
    intercept2 = scale // k if has_intercept2 else 0
    intercept3 = scale // l if has_intercept3 else 0

    return num_intercept, has_intercept1, has_intercept2, has_intercept3, intercept1, intercept2, intercept3


def get_vectors(
    num_intercept: int,
    has_intercept1: bool,
    has_intercept2: bool,
    has_intercept3: bool,
    intercept1: int,
    intercept2: int,
    intercept3: int
):

    vector1 = np.zeros(3, dtype = int)
    vector2 = np.zeros(3, dtype = int)
    vector3 = np.zeros(3, dtype = int)

    if num_intercept <= 1:
        setup_vectors1(
            has_intercept1, has_intercept2, has_intercept3,
            intercept1, intercept2, intercept3,
            vector1, vector2, vector3
        )
    elif num_intercept <= 2:
        setup_vectors2(
            has_intercept1, has_intercept2, has_intercept3, 
            intercept1, intercept2, intercept3, 
            vector1, vector2, vector3
        )
    else:
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


def setup_vectors3(
    intercept1: int,
    intercept2: int,
    intercept3: int,
    vector1: np.ndarray,
    vector2: np.ndarray,
    vector3: np.ndarray,
):

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


def get_boundary_box(
    vector1: np.ndarray,
    vector2: np.ndarray,
    vector3: np.ndarray
) -> np.ndarray:

    # bound_box = [[0, 0] for _ in range(3)]
    bound_box = np.zeros((3, 2), dtype = int)

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


def get_lattice_vecs(
    atoms: Atoms,
    vector1: np.ndarray,
    vector2: np.ndarray,
    vector3: np.ndarray
) -> np.ndarray:

    latt_int = np.stack([vector1, vector2, vector3])
    latt_vecs_bulk = atoms.cell[:]
    latt_unit0 = np.dot(latt_int, latt_vecs_bulk)
    latt_consts = get_cell_dm_14(latt_unit0)
    latt_vecs = get_cell_14(latt_consts)
    
    return latt_vecs


def get_converted_atoms(
    atoms: Atoms,
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
                        # atom = Atom(symbol, np.dot(shifted_abc, latt_vecs_new))
                        atom = SlabAtom(symbol, np.dot(shifted_abc, latt_vecs_new))

                        # print(f"add count {add_count}")
                        add_count += 1

                        atoms_set.add(atom)
    
    # converted_atoms = Atoms()
    # converted_atoms.set_cell(latt_vecs_new)
    # for atom in atoms_set:
    #     converted_atoms.append(atom)
    # print("-----")
    # print(f"Length: {len(list(atoms_set))}")
    # print("-----")
    converted_atoms = SlabBulk(latt_vecs_new, list(atoms_set), [])

    return converted_atoms


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


def get_slabs(lattice: SlabBulk, layers: int) -> Slab:
    """Create surfaces from a given lattice.
        lattice: Bulk lattice structure.
        layers: Number of equivalent layers of the slab.
    """
    # lattice_vecs = lattice.get_cell()
    trans_vec_set = lattice.trans_vec_set
    nstep = int(trans_vec_set[2][2] / STEP_FOR_GENOMS)
    slab_genoms = {}

    for i in range(nstep):
        offset = i / nstep
        slab_genom = get_slab_genom(offset, trans_vec_set, lattice.atoms)
        if slab_genom is not None and slab_genom not in slab_genoms:
            slab_genoms[slab_genom] = offset

    # slab_models = [SlabModelLeaf(self, offset) for offset in slab_genoms.values()]
    offsets = [offset for offset in slab_genoms.values()]
    lattice.setup_bonds()
    slabs = [lattice.to_slab(offset, layers) for offset in offsets]
    # new_atoms, new_latt = lattice.to_slab(offset, layers, adsorbates=None)

    return slabs


def get_slab_genom(offset, trans_vec_set, atoms):

    natom = len(trans_vec_set)
    iatom = natom

    names = []
    coords = []

    for atom in atoms:

        c1 = atom.get_coord_frac(trans_vec_set)[2] + offset
        c2 = c1 - int(c1)

        dc = abs(c2 - 1.0)
        dz = dc * trans_vec_set[2][2]
        if dz < POSIT_THR:
            c2 -= 1.0

        dz = abs(c1 - c2) * trans_vec_set[2][2]
        if iatom >= natom and dz < POSIT_THR:
            iatom = atoms.index(atom)

        names.append(atom.element)
        coords.append(c2 * trans_vec_set[2][2])   

    names2 = names[iatom:] + names[:iatom]
    coords2 = coords[iatom:] + coords[:iatom]

    return SlabGenom(names2, coords2)
    