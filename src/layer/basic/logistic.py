import theano
from theano import tensor

from theano_impl.theano_smart_layer import TheanoSmartLayer


class LogisticLayer(TheanoSmartLayer):

    def info(self):
        return [
            ("x", "input", ["samples", "features"]),
            ("W", "weight", ["features"]),
            ("b", "weight", [], "extra"),
            ("y", "output", ["samples"])
        ]

    def get_theano_output_smart(self, n):
        n.y = theano.tensor.nnet.sigmoid(
                theano.tensor.dot(n.x, n.W) + (n.b if n.b else 0)
        )
