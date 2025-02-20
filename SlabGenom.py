class SlabGenom():

    COORD_THR: float = 0.10 # angstrom
    LAYER_THR: float = 0.10 # angstrom

    def __init__(self, names, coords):
        if (not names) or len(names) < 1:
            raise ValueError("names is empty.")
        if (not coords) or len(coords) < 1:
            raise ValueError("coords is empty.")
        if len(names) != len(coords):
            raise ValueError("names.length != coords.length.")
        self.setup_layers(names, coords)


    def setup_layers(self, names, coords):

        self.layers = []
        istart: int = 0
        iend: int = 0
        coord1: float = 0.0
        coord2: float = 0.0

        while True:
            istart = iend
            iend = self.next_layer(istart, coords)
            if istart >= iend:
                break

            coord1 = coord2
            coord2 = 0.0
            for i in range(istart, iend):
                coord2 += coords[i]
            coord2 /= float(iend - istart)

            distance: float = 0.0
            if self.layers:
                distance = coord1 - coord2

            layer = self.get_layer(istart, iend, names)
            if layer:
                layer.distance = distance
                self.layers.append(layer)


    def next_layer(self, istart: int, coords) -> int:

        if istart >= len(coords):
            return len(coords)

        iend: int = len(coords)
        coord0: float = coords[istart]

        for i in range(istart + 1, len(coords)):
            coord = coords[i]
            if abs(coord - coord0) > SlabGenom.COORD_THR:
                iend = i
                break

        return iend


    def get_layer(self, istart, iend, names):

        if (iend - istart) == 1:
            layer = Layer()
            layer.code = names[istart]
            return layer

        names2 = names[istart:iend]
        names2.sort()

        mult = 0
        name = None
        code_parts = []

        for i in range(len(names2) + 1):
            if i < len(names2) and name == names2[i]:
                mult += 1
                continue

            if name is not None and name != "":
                if len(code_parts) > 0:
                    code_parts.append(' ')

                code_parts.append(name)
                if mult > 1:
                    code_parts.append(f'*{mult}')

            if i < len(names2):
                mult = 1
                name = names2[i]

        layer = Layer()
        layer.code = ''.join(code_parts)
        return layer
    

    def __str__(self):

        str_builder = []

        if self.layers is not None:
            for layer in self.layers:
                if layer is not None:
                    str_builder.append(f'{{{layer.code}|{layer.distance}}}')

        return ''.join(str_builder) if str_builder else '{}'
    

    def __hash__(self):

        return hash(tuple(self.layers)) if self.layers is not None else 0
    

    def __eq__(self, other):
        
        if self is other:
            return True
        
        if other is None or not isinstance(other, SlabGenom):
            return False
        
        if self.layers is None:
            return other.layers is None
        else:
            return self.layers == other.layers
    

class Layer:

    LAYER_THR = 1e-9

    def __init__(self, code=None, distance=0):
        self.code = code
        self.distance = distance

    def __hash__(self):
        return hash(self.code) if self.code is not None else 0

    def __eq__(self, other):
        if self is other:
            return True

        if other is None or not isinstance(other, Layer):
            return False

        if self.code != other.code:
            return False

        return abs(self.distance - other.distance) <= Layer.LAYER_THR
