import theano
from theano import tensor
from theano_impl.theano_smart_layer import TheanoSmartLayer


class LocalMatch(TheanoSmartLayer):

    def info(self):
        return [
            ("question", "input", ["samples", "length1", "features"]),
            ("question_mask", "input", ["samples", "length1"]),
            ("answer", "input", ["samples", "length2", "features"]),
            ("answer_mask", "input", ["samples", "length2"]),
            ("local_relevance", "output", ["samples", "length1", "length2"]),
            ("relevance", "output", ["samples"])
        ]

    def get_theano_output_smart(self, n):
        q = n.question.dimshuffle(1, 0, 2)
        q_mask = n.question_mask.dimshuffle(1, 0)
        a = n.answer.dimshuffle(1, 0, 2)
        a_mask = n.answer_mask.dimshuffle(1, 0)

        def loop_2(a_t, a_mask_t, q_t, q_mask_t):
            # q_t: samples * features
            d = tensor.sum(a_t * q_t, axis=1)
            l1 = tensor.sum(a_t * a_t, axis=1)
            l2 = tensor.sum(q_t * q_t, axis=1)
            return d / ((l1 * l2) ** 0.5)

        def loop_1(q_t, q_mask_t):
            return theano.scan(
                    loop_2,
                    sequences=[a, a_mask],
                    non_sequences=[q_t, q_mask_t],
                    outputs_info=[None])[0]

        n.local_relevance = theano.scan(
                loop_1,
                sequences=[q, q_mask],
                outputs_info=[None]
        )[0].dimshuffle(2, 0, 1)

        maximum_local_relevance = tensor.max(n.local_relevance, axis=2)
        n.relevance = tensor.sum(maximum_local_relevance, axis=1)
