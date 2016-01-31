import theano
from theano import tensor
from layer.smart_layer import SmartLayer


class SentenceConvolutionLayer(SmartLayer):

    def info(self):
        return [
            ("sentence", "input", ["samples", "features", "max_length"]),
            ("mask", "input", ["samples", "max_length"]),
            ("filter", "weight", ["depth", "features", "window"]),
            ("output", "output", ["samples", "max_length - window + 1", "depth"])
        ]

    def get_theano_output_smart(self, n):
        filter_shape = self.filter.get_shape()
        conv = theano.tensor.nnet.conv.conv2d(
            input=n.sentence.dimshuffle(0, 'x', 1, 2),
            filters=n.filter.dimshuffle(0, 'x', 1, 2),
            filter_shape=[filter_shape[0], filter_shape[1], 1, filter_shape[2]]
        )
        window = filter_shape[2]
        mask = n.mask[:, :1 - window]
        n.output = theano.tensor.nnet.sigmoid(
            theano.tensor.sum(conv, axis=2)
        ).dimshuffle(0, 2, 1) * mask.dimshuffle(0, 1, 'x')
