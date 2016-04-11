import theano
from theano import tensor
from theano_impl.theano_smart_layer import TheanoSmartLayer


class GruLayer(TheanoSmartLayer):

    def info(self):
        return [
            ("x", "input", ["samples", "length", "features"]),
            ("mask", "input", ["samples", "length"]),
            ("W_z", "weight", ["hidden + features", "hidden"]),
            ("b_z", "weight", [], "extra"),
            ("W_r", "weight", ["hidden + features", "hidden"]),
            ("b_r", "weight", [], "extra"),
            ("W_h", "weight", ["hidden + features", "hidden"]),
            ("b_h", "weight", [], "extra"),
            ("output", "output", ["samples", "length", "hidden"]),
            ("last", "output", ["samples", "hidden"]),
        ]

    def get_theano_output_smart(self, n):
        initial_value = theano.tensor.zeros_like(
            theano.tensor.dot(
                theano.tensor.sum(n.x, axis=[1, 2]).dimshuffle(0, 'x'),
                n.W_h
            )
        )
        x_r = n.x.dimshuffle(1, 0, 2)
        mask_r = n.mask.dimshuffle(1, 0)

        def step(x, mask, s_prev):
            x1 = tensor.concatenate([x, s_prev], axis=1)
            z_gate = tensor.nnet.sigmoid(
                tensor.dot(x1, n.W_z) + (n.b_z if n.b_z else 0.)
            )
            r_gate = tensor.nnet.sigmoid(
                tensor.dot(x1, n.W_r) + (n.b_r if n.b_r else 0.)
            )
            x2 = tensor.concatenate([x, r_gate * s_prev], axis=1)
            s_new = tensor.tanh(
                tensor.dot(x2, n.W_h) + (n.b_h if n.b_h else 0.)
            )
            s = (1 - z_gate) * s_prev + z_gate * s_new
            mask = mask.dimshuffle(0, 'x')
            final_s = mask * s + (1 - mask) * s_prev
            return final_s

        states, _ = theano.scan(
            fn=step,
            sequences=[x_r, mask_r],
            outputs_info=[initial_value]
        )
        n.output = states.dimshuffle(1, 0, 2)
        n.last = n.output[:, -1]
