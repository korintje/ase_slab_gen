
import numpy as np

class SlabModel():

    DEFAULT_OFFSET = 0.0
    DEFAULT_THICKNESS = 1.0
    DEFAULT_VACUUM = 10.0 # angstrom
    DEFAULT_SCALE = 1

    def __init__(self):
        self.offset: float = SlabModel.DEFAULT_OFFSET
        self.thickness: float = SlabModel.DEFAULT_THICKNESS
        self.vacuum: float = SlabModel.DEFAULT_VACUUM
        self.scaleA: int = SlabModel.DEFAULT_SCALE
        self.scaleB: int = SlabModel.DEFAULT_SCALE

    def set_offset(self, offset: float):
        self.offset = offset

    def get_offset(self) -> float:
        return self.offset

    def set_thickness(self, thickness: float):
        if thickness <= 0.0:
            raise ValueError(
                f"Thickness must be higher than 0.0, but {thickness} given."
            )
        else:
            self.thickness = thickness

    def get_thickness(self) -> float:
        return self.thickness

    def set_vacuum(self, vacuum: float):
        if vacuum < 0.0:
            raise ValueError(
                f"Vacuum thickness must be higher than 0.0, but {vacuum} given."
            )
        else:
            self.vacuum = vacuum

    def get_vacuum(self) -> float:
        return self.vacuum

    def set_scaleA(self, scaleA: int):
        if scaleA <= 0:
            raise ValueError(
                f"Scale factor A must be higher than 0, but {scaleA} given."
            )
        self.scaleA = scaleA

    def get_scaleA(self) -> int:
        return self.scaleA

    def set_scaleB(self, scaleB: int):
        if scaleB <= 0:
            raise ValueError(
                f"Scale factor B must be higher than 0, but {scaleB} given."
            )
        self.scaleB = scaleB

    def get_scaleB(self) -> int:
        return self.scaleB



    # def is_polar(self, tol_dipole_per_unit_area=1e-3) -> bool:
        """Checks whether the surface is polar by computing the dipole per unit
        area. Note that the Slab must be oxidation state-decorated for this
        to work properly. Otherwise, the Slab will always be non-polar.

        Args:
            tol_dipole_per_unit_area (float): A tolerance. If the dipole
                magnitude per unit area is less than this value, the Slab is
                considered non-polar. Defaults to 1e-3, which is usually
                pretty good. Normalized dipole per unit area is used as it is
                more reliable than using the total, which tends to be larger for
                slabs with larger surface areas.
        """
    #     dip_per_unit_area = self.dipole / self.surface_area
    #     return np.linalg.norm(dip_per_unit_area) > tol_dipole_per_unit_area


    # @property
    # def dipole(self):
        """Calculates the dipole of the Slab in the direction of the surface
        normal. Note that the Slab must be oxidation state-decorated for this
        to work properly. Otherwise, the Slab will always have a dipole of 0.
        """
    #     dipole = np.zeros(3)
    #     mid_pt = np.sum(self.cart_coords, axis=0) / len(self)
    #     normal = self.normal
    #     for site in self:
    #         charge = sum(getattr(sp, "oxi_state", 0) * amt for sp, amt in site.species.items())
    #         dipole += charge * np.dot(site.coords - mid_pt, normal) * normal
    #     return dipole
        


