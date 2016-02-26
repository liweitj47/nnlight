from utility.debug import NNDebug


class BackendBuilder:
    """
    An abstract class for backend builder objects
    """
    def __init__(self, core):
        self.core = core

    def get_constructor_map(self):
        return None

    def build(self):
        NNDebug.error("[BackendBuilder] build() method must be override")
