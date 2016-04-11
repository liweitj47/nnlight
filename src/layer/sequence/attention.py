import theano
from theano import tensor
from theano_impl.theano_smart_layer import TheanoSmartLayer


class AttentionGruLayer(TheanoSmartLayer):

    def info(self):
        return [
            ("x", "input", ["samples", "length1", "features1"]),
            ("mask", "input", ["samples", "length1"]),
            ("sequential_context", "input", ["samples", "length2", "features2"]),
            ("global_context", "input", ["samples", "features3"], "extra"),

            ("W_attention_context", "weight", ["features2"]),
            ("W_attention_state", "weight", ["hidden"]),
            ("b_attention", "weight", [], "extra"),

            ("W_z", "weight", ["hidden + features1 + features2 + features3=0", "hidden"]),
            ("b_z", "weight", [], "extra"),
            ("W_r", "weight", ["hidden + features1 + features2 + features3=0", "hidden"]),
            ("b_r", "weight", [], "extra"),
            ("W_h", "weight", ["hidden + features1 + features2 + features3=0", "hidden"]),
            ("b_h", "weight", [], "extra"),

            ("output", "output", ["samples", "length1", "hidden"]),
            ("last", "output", ["samples", "hidden"]),
            ("context", "output", ["samples", "length1", "features2"])
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
            attention = tensor.nnet.softmax(
                    tensor.tensordot(n.sequential_context, n.W_attention_context, [[2], [0]]) +
                    tensor.dot(s_prev, n.W_attention_state).dimshuffle(0, 'x') +
                    (n.b_attention if n.b_attention else 0.)
                )
            local_context = tensor.sum(attention.dimshuffle(0, 1, 'x') * n.sequential_context, axis=1)

            inputs = [x, s_prev, local_context]
            if n.global_context is not None:
                inputs.append(n.global_context)
            x1 = tensor.concatenate(inputs, axis=1)
            z_gate = tensor.nnet.sigmoid(
                tensor.dot(x1, n.W_z) + (n.b_z if n.b_z else 0.)
            )
            r_gate = tensor.nnet.sigmoid(
                tensor.dot(x1, n.W_r) + (n.b_r if n.b_r else 0.)
            )

            inputs = [x, r_gate * s_prev, local_context]
            if n.global_context is not None:
                inputs.append(n.global_context)
            x2 = tensor.concatenate(inputs, axis=1)
            s_new = tensor.tanh(
                tensor.dot(x2, n.W_h) + (n.b_h if n.b_h else 0.)
            )
            s = (1 - z_gate) * s_prev + z_gate * s_new

            mask = mask.dimshuffle(0, 'x')
            final_s = mask * s + (1 - mask) * s_prev
            final_c = mask * local_context
            return final_s, final_c

        (states, contexts), _ = theano.scan(
            fn=step,
            sequences=[x_r, mask_r],
            outputs_info=[initial_value, dict(initial=None)]
        )
        n.output = states.dimshuffle(1, 0, 2)
        n.last = n.output[:, -1]
        n.context = contexts.dimshuffle(1, 0, 2)
