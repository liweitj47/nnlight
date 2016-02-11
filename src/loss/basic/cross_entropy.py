import theano
from theano import tensor
from loss.loss import Loss
from layer.smart_layer import SmartLayer


class BinaryCrossEntropyLoss(Loss, SmartLayer):

    def __init__(self, name, params, core):
        SmartLayer.__init__(self, name, params, core)

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


class SequencialBinaryCrossEntropyLoss(Loss, SmartLayer):

    def __init__(self, name, params, core):
        SmartLayer.__init__(self, name, params, core)

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


class CrossEntropyLoss(Loss, SmartLayer):

    def __init__(self, name, params, core):
        SmartLayer.__init__(self, name, params, core)

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


class SequencialCrossEntropyLoss(Loss, SmartLayer):

    def __init__(self, name, params, core):
        SmartLayer.__init__(self, name, params, core)

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
