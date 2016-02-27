from layer.basic.input import InputLayer as InputLayerBase


class InputLayer(InputLayerBase):

    def get_computation_on_java_code(self, code, binder):
        code.assignment("this." + binder.get_name(self.get_value()), code.escape(self.name) )
