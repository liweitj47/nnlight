import theano
from theano import tensor

from loss.loss import Loss
from theano_impl.theano_smart_layer import TheanoSmartLayer


class BinaryCrossEntropyLoss(Loss, TheanoSmartLayer):

    def __init__(self, name, params, core):
        TheanoSmartLayer.__init__(self, name, params, core)  # this is necessary because multi-inheritance

    def info(self):
        return [
            ("predict", "input", ["samples"]),
            ("golden", "input", ["samples"]),
            ("loss", "output", [])
        ]

    def get_theano_output_smart(self, n):
        n.loss = -tensor.mean(
            tensor.xlogx.xlogy0(n.golden, n.predict) +
            tensor.xlogx.xlogy0(1-n.golden, 1-n.predict)
        )


class SequentialBinaryCrossEntropyLoss(Loss, TheanoSmartLayer):

    def __init__(self, name, params, core):
        TheanoSmartLayer.__init__(self, name, params, core)

    def info(self):
        return [
            ("predict", "input", ["samples, length"]),
            ("golden", "input", ["samples", "length"]),
            ("loss", "output", [])
        ]

    def get_theano_output_smart(self, n):
        n.loss = -theano.tensor.mean(
            theano.tensor.xlogx.xlogy0(n.golden, n.predict)
        )


class CrossEntropyLoss(Loss, TheanoSmartLayer):

    def __init__(self, name, params, core):
        TheanoSmartLayer.__init__(self, name, params, core)

    def info(self):
        return [
            ("predict", "input", ["samples", "labels"]),
            ("golden", "input", ["samples", "labels"]),
            ("loss", "output", [])
        ]

    def get_theano_output_smart(self, n):
        product = tensor.xlogx.xlogy0(n.golden, n.predict)
        n.loss = -tensor.mean(
            tensor.sum(product, axis=1)
        )


class SequentialCrossEntropyLoss(Loss, TheanoSmartLayer):

    def __init__(self, name, params, core):
        TheanoSmartLayer.__init__(self, name, params, core)

    def info(self):
        return [
            ("predict", "input", ["samples", "length", "labels"]),
            ("golden", "input", ["samples", "length", "labels"]),
            ("loss", "output", [])
        ]

    def get_theano_output_smart(self, n):
        product = tensor.xlogx.xlogy0(n.golden, n.predict)
        n.loss = -tensor.mean(
            tensor.sum(product, axis=2)
        )
