from theano import tensor

from loss.loss import Loss
from theano_impl.theano_smart_layer import TheanoSmartLayer


class MaxMarginLoss(Loss, TheanoSmartLayer):

    def __init__(self, name, params, core):
        TheanoSmartLayer.__init__(self, name, params, core)  # this is necessary because multi-inheritance
        self.margin = self.params.get("margin", 0.2)

    def info(self):
        return [
            ("positive", "input", ["samples"]),
            ("negative", "input", ["samples"]),
            ("loss", "output", []),
            ("error_count", "output", [], "int64")
        ]

    def get_theano_output_smart(self, n):
        if not isinstance(self.margin, float):
            self.error("margin should be of float type")

        diff = n.positive - n.negative
        n.loss = -tensor.mean(
            tensor.switch(
                tensor.lt(diff, self.margin), diff, 0.0
            )
        )
        n.error_count = tensor.sum(
            tensor.switch(
                tensor.lt(diff, self.margin), 1, 0
            )
        )
