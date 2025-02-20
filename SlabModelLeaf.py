from SlabModel import SlabModel

class SlabModelLeaf(SlabModel):


    def __init__(self, stem, offset):

        super().__init__()

        if stem is None:
            raise ValueError("stem is null.")

        self.stem = stem
        self.offset = offset


    def get_slab_models(self):
        return [self]


    def to_atoms(self):
        return self.stem.to_atoms(self)
