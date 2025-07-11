import numpy as np
from itertools import product
from ase import Atom as ASE_Atom
from ase import Atoms as ASE_Atoms

COEFFS = (0.0, 1.0, -1.0)
BOUND_BOXES = np.array(tuple(product(COEFFS, COEFFS, COEFFS)))
SCALE_FACTOR = 1.4
MAX_FOR_CTHICK: int = 20
POSIT_THR: float = 1.0e-4      # angstrom
STEP_FOR_CTHICK: float = 0.05  # internal coordinate
VEC_FACTOR: float = 1.0e-4
THICK_THR: float = 1.0e-12     # angstrom


class SlabBond():


    def __init__(
        self,
        tail_element: str,
        head_element: str,
        tail_coord: np.ndarray,
        head_coord: np.ndarray,
        distance: float
    ):

        self.tail_element: str = tail_element
        self.head_element: str = head_element
        self.tail_coord: np.ndarray = tail_coord
        self.head_coord: np.ndarray = head_coord
        self.distance: float = distance


    def get_tail_coord(self):

        return self.tail_coord


    def get_head_coord(self):

        return self.head_coord


    def get_tail_coord_frac(self, trans_vec_set: np.ndarray):

        inv_vec_set = np.linalg.inv(trans_vec_set)
        tail_coord_frac = np.dot(self.tail_coord, inv_vec_set)
        return tail_coord_frac

    
    def get_head_coord_frac(self, trans_vec_set: np.ndarray):

        inv_vec_set = np.linalg.inv(trans_vec_set)
        head_coord_frac = np.dot(self.head_coord, inv_vec_set)
        return head_coord_frac


class SlabAtom():

    POSIT_THR = 0.01

    def __init__(self, element, coord):

        self.element: str = element
        self.coord: np.ndarray = coord


    def __lt__(self, other):

        return self._compare_to_static(self, other) < 0


    def __eq__(self, other):
        
        if isinstance(other, SlabAtom):
            return self._equals_static(self, other)
        return False


    def __hash__(self):
        
        return hash(self.element)


    @staticmethod
    def _compare_to_static(atom1, atom2):

        if atom1 is None:
            return -1

        dxyz = atom1.get_coord() - atom2.get_coord()
        dx = dxyz[0]
        dy = dxyz[1]
        dz = dxyz[2]
        if dz > SlabAtom.POSIT_THR:
            if dz > 0.0:
                return 1
            else:
                return -1
        elif dy > SlabAtom.POSIT_THR:
            if dy > 0.0:
                return 1
            else:
                return -1
        elif dz > SlabAtom.POSIT_THR:
            if dx > 0.0:
                return 1
            else:
                return -1

        if atom1.element is None:
            return 1 if atom2.element is not None else 0

        return (atom1.element > atom2.element) - (atom1.element < atom2.element)


    @staticmethod
    def _equals_static(entry, obj):

        if entry is obj:
            return True
        if obj is None or not isinstance(obj, SlabAtom):
            return False

        other = obj
        if entry.element != other.element:
            return False

        dxyz = entry.get_coord() - other.get_coord()
        rr = np.dot(dxyz, dxyz)

        return rr <= SlabAtom.POSIT_THR ** 2
    
    
    def set_coord(self, coord: np.ndarray):

        self.coord = coord


    def get_coord(self):
        
        return self.coord

    
    def get_coord_frac(self, trans_vec_set: np.ndarray):
        
        inv_vec_set = np.linalg.inv(trans_vec_set)
        coord_frac = np.dot(self.coord, inv_vec_set)
        return coord_frac


    def get_inner_bond_to(self, end_atom):

        rel_coord = end_atom.coord - self.coord
        distance = np.linalg.norm(rel_coord)
        virtual_bond = SlabBond(
                self.element,
                end_atom.element,
                self.coord,
                self.coord + rel_coord,
                distance,
            )
        return virtual_bond


    def get_outer_bonds_to(self, end_atom, trans_vec_set: np.ndarray):
        
        trans_vec_sets = np.delete(BOUND_BOXES, 0, 0) @ trans_vec_set
        end_atom_coords = trans_vec_sets + end_atom.coord
        rel_coords = end_atom_coords - self.coord
        distances = np.linalg.norm(rel_coords, axis=1)
        virtual_bonds = [
            SlabBond(
                self.element,
                end_atom.element,
                self.coord,
                self.coord + rel_coord,
                distance,
            ) for rel_coord, distance in zip(rel_coords, distances)
        ]

        return virtual_bonds


class SlabBulk():
    

    def __init__(self, trans_vec_set, atoms, bonds):
        
        self.trans_vec_set = trans_vec_set
        self.atoms = atoms # List of Atom
        self.bonds = bonds # List of Bonds

    
    def setup_bonds(self):

        min_distance = np.inf
        virtual_bonds = []
        
        # Prepare virtual bonds with outer atoms
        atoms = self.atoms[:]
        for tail_atom in atoms:
            for head_atom in atoms:
                vbonds = tail_atom.get_outer_bonds_to(head_atom, self.trans_vec_set)
                virtual_bonds.extend(vbonds)

        # Prepare virtual bonds with inner atoms
        while atoms:
            tail_atom: SlabAtom = atoms.pop()
            for head_atom in atoms:
                vbond = tail_atom.get_inner_bond_to(head_atom)
                virtual_bonds.append(vbond)

        min_distance = min([vbond.distance for vbond in virtual_bonds])   
        thr_distance = SCALE_FACTOR * min_distance
        
        # Filer virtual bonds by the distance threashold to add bonds
        self.bonds = []
        for vbond in virtual_bonds:
            if vbond.distance < thr_distance:
                self.bonds.append(vbond)
 
    
    def to_slab(self, offset: float, thickness: int):

        thickness = int(thickness)

        new_trans_vec_set: np.ndarray = self.trans_vec_set * np.array([
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [float(thickness), float(thickness), float(thickness)]
        ])

        new_atoms = self.create_slab_atoms(offset, thickness)
        top_bonds, bottom_bonds = self.create_dangling_bonds(offset, thickness)

        return Slab(new_trans_vec_set, new_atoms, top_bonds, bottom_bonds)


    def create_slab_atoms(self, offset, thickness):

        new_atoms = []
        for i in range(thickness):
            bottom_atoms = []
            top_atoms = []
            for atom in self.atoms:
                abc = atom.get_coord_frac(self.trans_vec_set)
                c_shift = offset - np.floor(abc[2] + offset) + float(i)
                new_abc = abc + np.array([0.0, 0.0, c_shift])
                new_coord = new_abc @ self.trans_vec_set
                atom_group = bottom_atoms if c_shift < 0 else top_atoms
                atom_group.append(
                    SlabAtom(
                        atom.element,
                        new_coord,
                    )
                )
            new_atoms.extend(bottom_atoms + top_atoms)
        
        return new_atoms


    def create_dangling_bonds(self, offset, thickness):
        
        top_z_shift = float(thickness - 1) * self.trans_vec_set[2]
        top_bonds = []
        bottom_bonds = []
        for bond in self.bonds:
            
            abc_tail = bond.get_tail_coord_frac(self.trans_vec_set)
            abc_head = bond.get_head_coord_frac(self.trans_vec_set)
            bond_vec = bond.get_head_coord() - bond.get_tail_coord()

            shifted_tail = abc_tail + np.array([0.0, 0.0, offset])
            shifted_head = abc_head + np.array([0.0, 0.0, offset])

            # Tail is in, head is out -> vec for top and inv_vec for bottom
            if shifted_tail[2] < 1.0 and shifted_head[2] >= 1.0:
                inner_coord_tail = shifted_tail @ self.trans_vec_set
                outer_coord_head = inner_coord_tail + bond_vec
                if 0.0 <= shifted_tail[0] < 1.0 and 0.0 <= shifted_tail[1] < 1.0:
                    top_bonds.append(
                        SlabBond(
                            bond.tail_element,
                            bond.head_element,
                            inner_coord_tail + top_z_shift,
                            outer_coord_head + top_z_shift,
                            bond.distance,
                        )
                    )
                inner_coord_head = outer_coord_head - self.trans_vec_set[2]
                outer_coord_tail = inner_coord_head - bond_vec
                if 0.0 <= shifted_head[0] < 1.0 and 0.0 <= shifted_head[1] < 1.0:
                    bottom_bonds.append(
                        SlabBond(
                            bond.head_element,
                            bond.tail_element,
                            inner_coord_head,
                            outer_coord_tail,
                            bond.distance,
                        )
                    )
            
            # Tail is in, head is out -> vec for top and inv_vec for bottom
            elif 0.0 <= shifted_tail[2] < 1.0 and shifted_head[2] < 0.0:
                inner_coord_tail = shifted_tail @ self.trans_vec_set
                outer_coord_head = inner_coord_tail + bond_vec
                if 0.0 <= shifted_tail[0] < 1.0 and 0.0 <= shifted_tail[1] < 1.0:
                    bottom_bonds.append(
                        SlabBond(
                            bond.tail_element,
                            bond.head_element,
                            inner_coord_tail,
                            outer_coord_head,
                            bond.distance,
                        )
                    )
                outer_coord_tail = inner_coord_tail + self.trans_vec_set[2]
                inner_coord_head = outer_coord_tail + bond_vec
                if 0.0 <= shifted_head[0] < 1.0 and 0.0 <= shifted_head[1] < 1.0:
                    top_bonds.append(
                        SlabBond(
                            bond.head_element,
                            bond.tail_element,
                            inner_coord_head + top_z_shift,
                            outer_coord_tail + top_z_shift,
                            bond.distance,
                        )
                    )
            
            # Tail is out, head is in -> inv_vec for top and vec for bottom
            elif shifted_tail[2] >= 1.0 and shifted_head[2] < 1.0:
                inner_coord_head = shifted_head @ self.trans_vec_set
                outer_coord_tail = inner_coord_head - bond_vec
                if 0.0 <= shifted_head[0] < 1.0 and 0.0 <= shifted_head[1] < 1.0:
                    top_bonds.append(
                        SlabBond(
                            bond.head_element,
                            bond.tail_element,
                            inner_coord_head + top_z_shift,
                            outer_coord_tail + top_z_shift,
                            bond.distance,
                        )
                    )
                outer_coord_head = inner_coord_head - self.trans_vec_set[2]
                inner_coord_tail = outer_coord_head - bond_vec
                if 0.0 <= shifted_tail[0] < 1.0 and 0.0 <= shifted_tail[1] < 1.0:
                    bottom_bonds.append(
                        SlabBond(
                            bond.tail_element,
                            bond.head_element,
                            inner_coord_tail,
                            outer_coord_head,
                            bond.distance,
                        )
                    )

            # Tail is out, head is in -> inv_vec for top and vec for bottom
            elif shifted_tail[2] < 0.0 and 0.0 <= shifted_head[2] < 1.0:
                inner_coord_head = shifted_head @ self.trans_vec_set
                outer_coord_tail = inner_coord_head - bond_vec
                if 0.0 <= shifted_head[0] < 1.0 and 0.0 <= shifted_head[1] < 1.0:
                    bottom_bonds.append(
                        SlabBond(
                            bond.head_element,
                            bond.tail_element,
                            inner_coord_head,
                            outer_coord_tail,
                            bond.distance,
                        )
                    )
                outer_coord_head = inner_coord_head + self.trans_vec_set[2]
                inner_coord_tail = outer_coord_head - bond_vec
                if 0.0 <= shifted_tail[0] < 1.0 and 0.0 <= shifted_tail[1] < 1.0:
                    top_bonds.append(
                        SlabBond(
                            bond.tail_element,
                            bond.head_element,
                            inner_coord_tail + top_z_shift,
                            outer_coord_head + top_z_shift,
                            bond.distance,
                        )
                    )
        
        return top_bonds, bottom_bonds


class Slab():


    def __init__(self, trans_vec_set, atoms, top_bonds, bottom_bonds):
        
        self.trans_vec_set = trans_vec_set
        self.atoms = atoms # List of Atom
        self.top_bonds = top_bonds # List of Bonds
        self.bottom_bonds = bottom_bonds # List of Bonds


    def to_atoms(self, adsorbates=[]):

        top_ads_atoms = self.create_ads_atoms(self.top_bonds, adsorbates)
        bottom_ads_atoms = self.create_ads_atoms(self.bottom_bonds, adsorbates)
        all_atoms = self.atoms + top_ads_atoms + bottom_ads_atoms

        # return new_atoms, new_trans_vec_set
        ase_atoms = ASE_Atoms()
        ase_atoms.set_cell(self.trans_vec_set)
        ase_atoms.set_pbc((True, True, False))
        for atom in all_atoms:
            ase_atom = ASE_Atom(atom.element, atom.get_coord())
            ase_atoms.append(ase_atom)

        return ase_atoms


    def create_ads_atoms(self, bonds, adsorbates):

        ads_atoms = []
        for i, bond in enumerate(bonds):
            
            adsorbate_props = next((ads for ads in adsorbates if ads["on"] == bond.tail_element), None)
            if not adsorbate_props:
                continue
            
            adsorbate = adsorbate_props.get("adsorbate")
            if type(adsorbate) == ASE_Atoms:
                pass
            else:
                if type(adsorbate) == str:
                    ads_element = adsorbate
                elif type(adsorbate) == ASE_Atom:
                    ads_element = adsorbate.symbol
                bond_length = adsorbate_props.get("bond_length")
                if bond_length:
                    tail_coord = bond.get_tail_coord()
                    head_coord = bond.get_head_coord()
                    vec = head_coord - tail_coord
                    head_coord = bond_length / np.linalg.norm(vec) * vec + tail_coord
                    head_abc = head_coord @ np.linalg.inv(self.trans_vec_set)
                else:
                    head_abc = bond.get_head_coord_frac(self.trans_vec_set)
                a = head_abc[0]
                b = head_abc[1]
                da = - np.floor(a)
                db = - np.floor(b)
                ads_atom = SlabAtom(
                    ads_element,
                    np.array([
                        a + da, 
                        b + db, 
                        head_abc[2]
                    ]) @ self.trans_vec_set
                )
                ads_atoms.append(ads_atom)

        return ads_atoms