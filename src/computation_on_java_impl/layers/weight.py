from layer.basic.weight import WeightLayer as WeightLayerBase


class WeightLayer(WeightLayerBase):

    def get_computation_on_java_code(self, code, binder):
        code.returnn()

