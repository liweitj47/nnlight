import theano
from theano import tensor

from theano_impl.theano_smart_layer import TheanoSmartLayer


class LstmLayer(TheanoSmartLayer):

    def info(self):
        return [
            ("x", "input", ["samples", "features", "length"]),
            ("mask", "input", ["samples", "length"]),
            ("W_i", "weight", ["hidden+features", "hidden"]),
            ("b_i", "weight", [], "extra"),
            ("W_o", "weight", ["hidden+features", "hidden"]),
            ("b_o", "weight", [], "extra"),
            ("W_f", "weight", ["hidden+features", "hidden"]),
            ("b_f", "weight", [], "extra"),
            ("W_c", "weight", ["hidden+features", "hidden"]),
            ("b_c", "weight", [], "extra"),
            ("output", "output", ["samples", "length", "hidden"]),
            ("last", "output", ["samples", "hidden"])
        ]

    def get_theano_output_smart(self, n):
        initial_value = theano.tensor.zeros_like(
            theano.tensor.dot(
                theano.tensor.sum(n.x, axis=[1, 2]).dimshuffle(0, 'x'),
                n.W_c
            )
        )  # samples * hidden
        x_t = n.x.dimshuffle(1, 0, 2)
        mask_t = n.mask.dimshuffle(1, 0)

        def _step(vec, mask, h, c):

            state = theano.tensor.concatenate([vec, h], axis=1)
            mask = mask.dimshuffle(0, 'x')

            forget_gate = theano.tensor.nnet.sigmoid(
                theano.tensor.dot(state, n.W_f) + (n.b_f if n.b_f else 0.)
            )

            input_gate = theano.tensor.nnet.sigmoid(
                theano.tensor.dot(state, n.W_i) + (n.b_i if n.b_i else 0.)
            )

            c2 = theano.tensor.tanh(
                theano.tensor.dot(state, n.W_c) + (n.b_c if n.b_c else 0.)
            )

            c2 = forget_gate * c + input_gate * c2

            output_gate = theano.tensor.nnet.sigmoid(
                theano.tensor.dot(state, n.W_o) + (n.b_o if n.b_o else 0.)
            )

            h2 = output_gate * theano.tensor.tanh(c2)

            final_h = mask*h2 + (1-mask)*h
            final_c = mask*c2 + (1-mask)*c

            return final_h, final_c

        outputs, updates = theano.scan(
            _step,
            sequences=[x_t, mask_t],
            outputs_info=[initial_value, initial_value],
        )

        n.output = outputs[0].dimshuffle(1, 0, 2)  # mask.dimshuffle(0, 1, 'x')
        n.last = n.output[:, -1]
