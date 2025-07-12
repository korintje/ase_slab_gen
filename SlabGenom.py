# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 Takuro Hosomi
# Email: t.hosomi1410@gmail.com
# GitHub: https://github.com/korintje
#
# This script is licensed under the GNU Lesser General Public License v2.1 (LGPL-2.1).
# You should have received a copy of the GNU Lesser General Public License along with this script.
# If not, see <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>.
#
# The core slab generation logic is originally developed by nisihara.burai@gmail.com.
# GitHub: https://github.com/BURAI-team/burai


class Layer:
    """
    Represents a single atomic layer in a slab, identified by its chemical composition 
    and vertical distance from the previous layer.
    """
    LAYER_THR = 1e-9  # Tolerance for comparing distances (angstrom)

    def __init__(self, code=None, distance=0.0):
        self.code = code
        self.distance = distance

    def __hash__(self):
        return hash(self.code) if self.code is not None else 0

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, Layer):
            return False
        return (
            self.code == other.code
            and abs(self.distance - other.distance) <= Layer.LAYER_THR
        )


class SlabGenom:
    """
    Represents the vertical structure of a slab by capturing the sequence
    of atomic layers and their compositions and spacings.
    """
    COORD_THR = 0.10  # Tolerance for z-coordinate grouping (angstrom)
    LAYER_THR = 0.10  # Minimum spacing to recognize as a new layer (angstrom)

    def __init__(self, names, coords):
        if not names:
            raise ValueError("Input 'names' is empty.")
        if not coords:
            raise ValueError("Input 'coords' is empty.")
        if len(names) != len(coords):
            raise ValueError("Length mismatch between 'names' and 'coords'.")

        self.layers = []
        self._build_layers(names, coords)

    def _build_layers(self, names, coords):
        """
        Groups atoms into layers based on z-coordinate proximity and
        stores chemical composition for each layer.
        """
        i_start = 0
        i_end = 0
        prev_center = 0.0

        while True:
            i_start = i_end
            i_end = self._find_next_layer(i_start, coords)

            if i_start >= i_end:
                break

            # Compute layer center z
            layer_center = sum(coords[i_start:i_end]) / (i_end - i_start)

            # Compute interlayer distance
            distance = prev_center - layer_center if self.layers else 0.0
            prev_center = layer_center

            # Create and store layer
            layer = self._make_layer(names[i_start:i_end])
            if layer:
                layer.distance = distance
                self.layers.append(layer)

    def _find_next_layer(self, i_start, coords):
        """
        Finds the end index of the next layer starting from i_start.
        Groups atoms based on proximity in z-direction.
        """
        if i_start >= len(coords):
            return len(coords)

        z_ref = coords[i_start]

        for i in range(i_start + 1, len(coords)):
            if abs(coords[i] - z_ref) > self.COORD_THR:
                return i

        return len(coords)

    def _make_layer(self, names):
        """
        Generates a Layer object representing the composition of atoms in the layer.
        Atoms are sorted and grouped by element symbol.
        """
        if len(names) == 1:
            return Layer(code=names[0])

        sorted_names = sorted(names)
        code_parts = []

        current = None
        count = 0

        for name in sorted_names + [None]:
            if name == current:
                count += 1
            else:
                if current:
                    code_parts.append(current)
                    if count > 1:
                        code_parts.append(f"*{count}")
                    code_parts.append(" ")
                current = name
                count = 1

        return Layer(code=''.join(code_parts).strip())

    def __str__(self):
        return ''.join(
            f'{{{layer.code}|{layer.distance:.3f}}}' for layer in self.layers if layer
        ) or '{}'

    def __hash__(self):
        return hash(tuple(self.layers)) if self.layers else 0

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, SlabGenom):
            return False
        return self.layers == other.layers
