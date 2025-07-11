from ase import Atoms as ASE_Atoms
import numpy as np
from LatticeTools import get_cell_14, get_cell_dm_14


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
