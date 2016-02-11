from layer.smart_layer import Layer


class Loss(Layer):

    def __init__(self, name, params, core):
        Layer.__init__(self,  name, params, core)
