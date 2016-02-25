from layer.layers import Layer


class Loss(Layer):

    def __init__(self, name, params, core):
        Layer.__init__(self,  name, params, core)
