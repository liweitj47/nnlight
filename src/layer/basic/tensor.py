from theano import tensor

from theano_impl.theano_smart_layer import TheanoSmartLayer


class LowRankTensorLayer(TheanoSmartLayer):

    def info(self):
        return [
            ("left", "input", ["samples", "features1"]),
            ("right", "input", ["samples", "features2"]),
            ("W_l", "weight", ["hidden", "features1", "rank"]),
            ("W_r", "weight", ["hidden", "features2", "rank"]),
            ("output", "output", ["samples", "hidden"])
        ]

    def get_theano_output_smart(self, n):
        h_l = tensor.tensordot(n.left, n.W_l, [[1], [1]])
        h_r = tensor.tensordot(n.right, n.W_r, [[1], [1]])
        n.output = tensor.nnet.sigmoid(
                        tensor.sum(h_l * h_r, axis=2)
                    )
