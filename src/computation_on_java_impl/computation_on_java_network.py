from core.network import Network


class ComputationOnJavaNetwork(Network):

    def __init__(self, code):
        self.__code = code

    def code(self):
        return self.__code

    def dump(self, path):
        f = open(path, "w")
        f.write(self.code())
        f.close()
