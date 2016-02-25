import theano
from theano import tensor

from theano_impl.theano_smart_layer import TheanoSmartLayer


class SoftmaxLayer(TheanoSmartLayer):

    def info(self):
        return [
            ("x", "input", ["samples", "features"]),
            ("W", "weight", ["features", "labels"]),
            ("y", "output", ["samples", "labels"])
        ]

    def get_theano_output_smart(self, n):
        n.y = theano.tensor.nnet.softmax(
            theano.tensor.dot(n.x, n.W)
        )


class SequencialSoftmaxLayer(TheanoSmartLayer):

    def info(self):
        return [
            ("x", "input", ["samples", "length", "features"]),
            ("W", "weight", ["features", "labels"]),
            ("y", "output", ["samples", "length", "labels"])
        ]

    def get_theano_output_smart(self, n):
        n.y, _ = theano.scan(
            fn=lambda x: theano.tensor.nnet.softmax(
                             theano.tensor.dot(x, n.W)
                           ),
            sequences=[n.x]
        )
