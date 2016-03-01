from loss.basic.max_margin import MaxMarginLoss as MaxMarginLossBase


class MaxMarginLoss(MaxMarginLossBase):

    def get_computation_on_java_code(self, code, binder):
        datatype = binder.get_base_type(self.positive)
        x = binder.get_name(self.positive)
        y = binder.get_name(self.negative)
        loss = binder.get_name(self.loss)
        count = binder.get_name(self.error_count)
        code.field("int", "samples", val="%s.length" % x)
        code.assignment(loss, 0)
        code.assignment(count, 0)
        code.begin_for("int i=0; i<samples; i++")
        code.field(datatype, "diff", val="%s[i] - %s[i]" % (x, y))
        code.begin_if("diff < %f" % self.margin)
        code.assignment(loss, "diff", operator="+=")
        code.assignment(loss, 1, operator="+=")
        code.end()
        code.assignment(loss, "%s / samples" % loss)
        code.end()
