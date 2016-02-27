from layer.basic.tensor import LowRankTensorLayer as LowRankTensorLayerBase


class LowRankTensorLayer(LowRankTensorLayerBase):

    def info(self):
        return [
            ("left", "input", ["samples", "features1"]),
            ("right", "input", ["samples", "features2"]),
            ("W_l", "weight", ["hidden", "features1", "rank"]),
            ("W_r", "weight", ["hidden", "features2", "rank"]),
            ("output", "output", ["samples", "hidden"]),
            # intermediate results
            ("left_h", "output", ["rank"]),
            ("right_h", "output", ["rank"])
        ]

    def get_computation_on_java_code(self, code, binder):
        basetype = binder.get_base_type(self.left)
        code.field("int", "samples", val=binder.get_name(self.left) + ".length")
        code.field("int", "features1", val=binder.get_name(self.left) + "[0].length")
        code.field("int", "features2", val=binder.get_name(self.right) + "[0].length")
        code.field("int", "rank", val=binder.get_name(self.left_h) + ".length")
        code.field("int", "hidden", val=binder.get_name(self.output) + "[0].length")
        code.begin_for("int i=0; i<samples; i++")
        code.begin_for("int j=0; j<hidden; j++")
        code.begin_for("int k=0; k<rank; k++")
        code.assignment(binder.get_name(self.left_h) + "[k]", 0)
        code.begin_for("int l=0; l<features1; l++")
        code.assignment(binder.get_name(self.left_h) + "[k]",
                        "%s[i][l] * %s[j][l][k]" % (binder.get_name(self.left), binder.get_name(self.W_l)),
                        operator="+=")
        code.end()
        code.assignment(binder.get_name(self.right_h) + "[k]", 0)
        code.begin_for("int l=0; l<features2; l++")
        code.assignment(binder.get_name(self.right_h) + "[k]",
                        "%s[i][l] * %s[j][l][k]" % (binder.get_name(self.right), binder.get_name(self.W_r)),
                        operator="+=")
        code.end()
        code.end()
        code.assignment(binder.get_name(self.output) + "[i][j]", 0)
        code.begin_for("int k=0; k<rank; k++")
        code.assignment(binder.get_name(self.output) + "[i][j]",
                        "%s[k] * %s[k]" % (binder.get_name(self.left_h), binder.get_name(self.right_h)),
                        operator="+=")
        code.end()
        code.assignment(binder.get_name(self.output) + "[i][j]",
                        "1 / (1 + Math.exp(-%s)" % binder.get_name(self.output) + "[i][j]")
        code.end()
        code.end()
