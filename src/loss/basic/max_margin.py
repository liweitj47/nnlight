from theano import tensor
from loss.loss import Loss
from layer.smart_layer import SmartLayer


class MaxMarginLoss(Loss, SmartLayer):

    def info(self):
        return [
            ("positive", "input", ["samples"]),
            ("negative", "input", ["samples"]),
            ("loss", "output", []),
            ("error_count", "output", [], "int64")
        ]

    def get_theano_output_smart(self, n):
        margin = self.params.get("margin", 0.2)
        if not isinstance(margin, float):
            self.error("margin should be of float type")

        diff = n.positive - n.negative
        n.loss = -tensor.mean(
            tensor.switch(
                tensor.lt(diff, margin), diff, 0.0
            )
        )
        n.error_count = tensor.sum(
            tensor.switch(
                tensor.lt(diff, margin), 1, 0
            )
        )
