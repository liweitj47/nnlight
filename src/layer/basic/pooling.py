import theano
from theano import tensor
from layer.smart_layer import SmartLayer


class MaxPoolingWithTimeLayer(SmartLayer):

    def info(self):
        return [
            ("input", "input", ["samples", "duration", "features"]),
            ("output", "output", ["samples", "features"])
        ]

    def get_theano_output_smart(self, n):
        n.output = theano.tensor.max(n.input, axis=1)
