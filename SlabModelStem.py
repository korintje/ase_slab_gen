from ase import Atoms as ASE_Atoms
import numpy as np
from LatticeTools import get_cell_14, get_cell_dm_14
from SlabModel import SlabModel


class AtomEntry:
    
    POSIT_THR = 0.01

    def __init__(self, name, lattice_vecs: np.ndarray):

        self.lattice: np.ndarray = lattice_vecs
        self.name: str = name
        self.abc: np.ndarray = np.zeros(3, dtype = float)
        self.xyz: np.ndarray = np.zeros(3, dtype = float)

    def __lt__(self, other):
        return self._compare_to_static(self, other) < 0


    def __eq__(self, other):
        if isinstance(other, AtomEntry):
            return self._equals_static(self, other)
        return False

    def __hash__(self):
        return hash(self.name)

    @staticmethod
    def _compare_to_static(entry1, entry2):
        if entry2 is None:
            return -1

        da, db, dc = entry1.abc - entry2.abc
        dxyz = dc * entry1.lattice[2]
        rr = np.sum(dxyz ** 2)
        if rr > AtomEntry.POSIT_THR ** 2:
            return -1 if dc > 0.0 else 1

        dxyz = db * entry1.lattice[1]
        rr = np.sum(dxyz ** 2)
        if rr > AtomEntry.POSIT_THR ** 2:
            return 1 if db > 0.0 else -1

        dxyz = db * entry1.lattice[0]
        rr = np.sum(dxyz ** 2)
        if rr > AtomEntry.POSIT_THR ** 2:
            return 1 if da > 0.0 else -1

        if entry1.name is None:
            return 1 if entry2.name is not None else 0

        return (entry1.name > entry2.name) - (entry1.name < entry2.name)


    @staticmethod
    def _equals_static(entry, obj):
        if entry is obj:
            return True
        if obj is None or not isinstance(obj, AtomEntry):
            return False

        other = obj
        if entry.name != other.name:
            return False

        ea, eb, ec = entry.abc
        oa, ob, oc = other.abc

        da = min(abs(ea - oa), abs(ea - oa + np.copysign(0.5 - ea, 1)))
        db = min(abs(eb - ob), abs(eb - ob + np.copysign(0.5 - eb, 1)))
        dc = min(abs(ec - oc), abs(ec - oc + np.copysign(0.5 - ec, 1)))

        dabc = np.array([da, db, dc])
        dxyz = np.dot(dabc, entry.lattice)
        rr = np.dot(dxyz, dxyz)

        return rr <= AtomEntry.POSIT_THR ** 2


class SlabModelStem(SlabModel):

    DET_THR: float = 1.0e-8
    PACK_THR: float = 1.0e-6       # internal coordinate
    OFFSET_THR: float = 1.0e-12    # angstrom
    THICK_THR: float = 1.0e-12     # angstrom
    POSIT_THR: float = 1.0e-4      # angstrom
    VALUE_THR: float = 1.0e-12
    STEP_FOR_GENOMS: float = 0.50  # angstrom
    STEP_FOR_CTHICK: float = 0.05  # internal coordinate
    MAX_FOR_CTHICK: int = 20
    SLAB_FIX_THR: float = 0.1      # angstrom
    SLAB_FIX_RATE: float = 0.5     # internal coordinate


    def __init__(self, ase_atoms: ASE_Atoms, h: int, k: int, l: int):

        super().__init__()

        positions: np.ndarray = ase_atoms.get_positions()
        if positions.size == 0:
            raise ValueError("Given atoms object is blank.")

        self.setup_millers(ase_atoms, h, k, l)
        self.setup_unit_atoms_in_cell(ase_atoms)
        self.setup_unit_atoms_in_slab()

        self.latt_auxi = None   # List<List<float>>
        self.latt_slab = None   # List<List<float>>
        self.entry_auxi = []    # List<AtomEntry>
        self.entry_slab = []    # List<AtomEntry>


    def setup_unit_atoms_in_cell(self, ase_atoms: ASE_Atoms):

        positions: np.ndarray = ase_atoms.get_positions()
        symbols: np.ndarray = ase_atoms.get_chemical_symbols()
        lattice_vecs: np.ndarray = ase_atoms.get_cell()
        rec_lattice = np.linalg.inv(lattice_vecs)
        self.entry_unit = []
        for position, name in zip(positions, symbols):
            entry = AtomEntry(name, lattice_vecs)
            entry.abc = np.dot(position, rec_lattice)
            self.entry_unit.append(entry)


    def setup_unit_atoms_in_slab(self):

        latt_int = np.stack([self.vector1, self.vector2, self.vector3])
        inv_latt = np.linalg.inv(latt_int)

        unit_atoms_set = set()

        for ia in range(self.bound_box[0][0], self.bound_box[0][1] + 1):
            for ib in range(self.bound_box[1][0], self.bound_box[1][1] + 1):
                for ic in range(self.bound_box[2][0], self.bound_box[2][1] + 1):
                    for entry in self.entry_unit:
                        abc = np.dot(entry.abc + np.array([ia, ib, ic]), inv_latt)
                        if (-SlabModelStem.PACK_THR <= abc[0] < 1.0 + SlabModelStem.PACK_THR and 
                            -SlabModelStem.PACK_THR <= abc[1] < 1.0 + SlabModelStem.PACK_THR and 
                            -SlabModelStem.PACK_THR <= abc[2] < 1.0 + SlabModelStem.PACK_THR):
                            atom = AtomEntry(entry.name, self.latt_unit)
                            shifted_abc = abc - np.floor(abc)
                            dc = 1.0 - shifted_abc[2]
                            dz = dc * self.latt_unit[2][2]
                            if dz < SlabModelStem.POSIT_THR:
                                shifted_abc[2] -= 1.0
                            atom.abc = shifted_abc
                            unit_atoms_set.add(atom)
        
        unit_atoms = list(unit_atoms_set)
        self.entry_unit = sorted(unit_atoms)


    def setup_millers(self, ase_atoms: ASE_Atoms, h, k, l):

        positions: np.ndarray = ase_atoms.get_positions()
        if positions.size == 0:
            raise ValueError("Given atoms is blank.")

        if h == 0 and k == 0 and l == 0:
            raise ValueError("Miller indices [0, 0, 0] is not allowed.")

        self.miller1 = h
        self.miller2 = k
        self.miller3 = l

        self.setup_intercepts()
        self.setup_vectors()
        self.setup_boundary_box()
        self.setup_lattice(ase_atoms)


    def setup_intercepts(self):
        scale_min = 1
        scale_max = 1
        self.num_intercept = 0

        if self.miller1 != 0:
            scale_min = max(scale_min, abs(self.miller1))
            scale_max *= abs(self.miller1)
            self.num_intercept += 1
            self.has_intercept1 = True
        else:
            self.has_intercept1 = False

        if self.miller2 != 0:
            scale_min = max(scale_min, abs(self.miller2))
            scale_max *= abs(self.miller2)
            self.num_intercept += 1
            self.has_intercept2 = True
        else:
            self.has_intercept2 = False

        if self.miller3 != 0:
            scale_min = max(scale_min, abs(self.miller3))
            scale_max *= abs(self.miller3)
            self.num_intercept += 1
            self.has_intercept3 = True
        else:
            self.has_intercept3 = False

        if scale_min < 1:
            raise ValueError("scaleMin is not positive.")

        if scale_max < scale_min:
            raise ValueError("scaleMax < scaleMin.")

        if self.num_intercept < 1:
            raise ValueError("there are no intercepts.")

        scale = 0
        for i in range(scale_min, scale_max + 1):
            if self.has_intercept1 and (i % self.miller1) != 0:
                continue
            if self.has_intercept2 and (i % self.miller2) != 0:
                continue
            if self.has_intercept3 and (i % self.miller3) != 0:
                continue

            scale = i
            break

        if scale < 1:
            raise ValueError("cannot detect scale.")

        self.intercept1 = scale // self.miller1 if self.has_intercept1 else 0
        self.intercept2 = scale // self.miller2 if self.has_intercept2 else 0
        self.intercept3 = scale // self.miller3 if self.has_intercept3 else 0


    def setup_vectors(self):

        self.vector1: np.ndarray = np.zeros(3, dtype = int)
        self.vector2: np.ndarray = np.zeros(3, dtype = int)
        self.vector3: np.ndarray = np.zeros(3, dtype = int)

        if self.num_intercept <= 1:
            self.setup_vectors1()
        elif self.num_intercept <= 2:
            self.setup_vectors2()
        else:
            self.setup_vectors3()


    def setup_vectors1(self):
        if self.has_intercept1:
            if self.intercept1 > 0:
                self.vector1[1] = 1
                self.vector2[2] = 1
                self.vector3[0] = 1
            else:
                self.vector1[2] = 1
                self.vector2[1] = 1
                self.vector3[0] = -1

        elif self.has_intercept2:
            if self.intercept2 > 0:
                self.vector1[2] = 1
                self.vector2[0] = 1
                self.vector3[1] = 1
            else:
                self.vector1[0] = 1
                self.vector2[2] = 1
                self.vector3[1] = -1

        elif self.has_intercept3:
            if self.intercept3 > 0:
                self.vector1[0] = 1
                self.vector2[1] = 1
                self.vector3[2] = 1
            else:
                self.vector1[1] = 1
                self.vector2[0] = 1
                self.vector3[2] = -1


    def setup_vectors2(self):
        if not self.has_intercept3:  # cat in A-B plane
            sign1 = int((self.intercept1 > 0) - (self.intercept1 < 0))
            sign2 = int((self.intercept2 > 0) - (self.intercept2 < 0))
            self.vector1[2] = sign1 * sign2
            self.vector2[0] = self.intercept1
            self.vector2[1] = -self.intercept2
            self.vector3[0] = sign1
            self.vector3[1] = sign2

        elif not self.has_intercept2:  # cat in A-C plane
            sign1 = int((self.intercept1 > 0) - (self.intercept1 < 0))
            sign3 = int((self.intercept3 > 0) - (self.intercept3 < 0))
            self.vector1[1] = sign1 * sign3
            self.vector2[0] = -self.intercept1
            self.vector2[2] = self.intercept3
            self.vector3[0] = sign1
            self.vector3[2] = sign3

        elif not self.has_intercept1:  # cat in B-C plane
            sign2 = int((self.intercept2 > 0) - (self.intercept2 < 0))
            sign3 = int((self.intercept3 > 0) - (self.intercept3 < 0))
            self.vector1[0] = sign2 * sign3
            self.vector2[1] = self.intercept2
            self.vector2[2] = -self.intercept3
            self.vector3[1] = sign2
            self.vector3[2] = sign3


    def setup_vectors3(self):
        sign1 = (self.intercept1 > 0) - (self.intercept1 < 0)
        sign2 = (self.intercept2 > 0) - (self.intercept2 < 0)
        sign3 = (self.intercept3 > 0) - (self.intercept3 < 0)

        if sign3 > 0:
            self.vector1[1] = sign1 * self.intercept2
            self.vector1[2] = -sign1 * self.intercept3
            self.vector2[0] = -sign2 * self.intercept1
            self.vector2[2] = sign2 * self.intercept3
        else:
            self.vector1[0] = -sign1 * self.intercept1
            self.vector1[2] = sign1 * self.intercept3
            self.vector2[1] = sign2 * self.intercept2
            self.vector2[2] = -sign2 * self.intercept3

        self.vector3[0] = sign1
        self.vector3[1] = sign2
        self.vector3[2] = sign3


    def setup_boundary_box(self):

        self.bound_box = [[0, 0] for _ in range(3)]

        for i in range(3):
            if self.vector1[i] < 0:
                self.bound_box[i][0] += self.vector1[i]
            else:
                self.bound_box[i][1] += self.vector1[i]

            if self.vector2[i] < 0:
                self.bound_box[i][0] += self.vector2[i]
            else:
                self.bound_box[i][1] += self.vector2[i]

            if self.vector3[i] < 0:
                self.bound_box[i][0] += self.vector3[i]
            else:
                self.bound_box[i][1] += self.vector3[i]


    def setup_lattice(self, ase_atoms: ASE_Atoms):

        latt_int = np.stack([self.vector1, self.vector2, self.vector3])
        lattice: np.ndarray = ase_atoms.cell[:]
        latt_unit0 = np.dot(latt_int, lattice)
        self.latt_const = None if latt_unit0.size == 0 else get_cell_dm_14(latt_unit0)
        if self.latt_const is None or len(self.latt_const) < 6:
            raise ValueError("Lattice constants are invalid.")

        self.latt_unit = get_cell_14(self.latt_const)
        if self.latt_unit is None or self.latt_unit.size == 0 or len(self.latt_unit) < 3:
            raise ValueError("Lattice vectors are invalid.")
        for i in range(3):
            if self.latt_unit[i] is None or len(self.latt_unit[i]) < 3:
                raise ValueError(f"Lattice vector {i} is invalid.")
