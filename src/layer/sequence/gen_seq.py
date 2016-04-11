import theano
from theano import tensor
from theano_impl.theano_smart_layer import TheanoSmartLayer


class SequentialGeneratorWithContext(TheanoSmartLayer):

    def info(self):
        return [
            ("x", "input", ["samples", "length", "symbols"]),
            ("mask", "input", ["samples", "length"]),
            ("context", "input", ["samples", "length", "features"]),
            ("W", "weight", ["features + symbols", "symbols"]),
            ("y", "output", ["samples", "length", "symbols"])
        ]

    def get_theano_output_smart(self, n):
        initial_value = tensor.zeros_like(
            tensor.dot(
                tensor.sum(n.x, axis=[1, 2]).dimshuffle(0, 'x'),
                n.W
            )
        )
        x_r = n.x.dimshuffle(1, 0, 2)
        context_r = n.context.dimshuffle(1, 0, 2)
        mask_r = n.mask.dimshuffle(1, 0)

        def step(mask, c, x):
            mask = mask.dimshuffle(0, 'x')
            prediction = tensor.nnet.softmax(
                tensor.dot(tensor.concatenate([x, c], axis=1), n.W)
            ) * mask
            return prediction, x
        output, _ = theano.scan(
            fn=step,
            sequences=[mask_r, context_r],
            outputs_info=[dict(intial=None), initial_value]
        )
        n.y = output[0].dimshuffle(1, 0, 2)
