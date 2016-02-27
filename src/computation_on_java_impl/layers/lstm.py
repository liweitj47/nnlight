from layer.basic.lstm import LstmLayer as LstmLayerBase


class LstmLayer(LstmLayerBase):

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
            ("last", "output", ["samples", "hidden"]),
            # java computation buffer
            ("union_buf", "output", ["hidden+features"]),
            ("input_gate_buf", "output", ["hidden"]),
            ("output_gate_buf", "output", ["hidden"]),
            ("forget_gate_buf", "output", ["hidden"]),
            ("cell_buf", "output", ["hidden"])
        ]

    # global variable names:
    # sample index = "s"
    # step index = "t"
    # input data = "x", "mask"
    # output data = "h"
    # buf data = "buf", "input_gate", "output_gate", "forget_gate", "cell"
    # shape param = "hidden", "features"
    def get_computation_on_java_code(self, code, binder):
        self.data_code(code, binder)
        code.begin_for("int s=0; s<x.length; s++")
        self.per_sample_code(code, binder)
        code.end()

    def data_code(self, code, binder):
        table = [
            ("x", self.x),
            ("mask", self.mask),
            ("h", self.output),
            ("buf", self.union_buf),
            ("input_gate", self.input_gate_buf),
            ("output_gate", self.output_gate_buf),
            ("forget_gate", self.forget_gate_buf),
            ("cell", self.cell_buf)
        ]
        for alias, value in table:
            typ, name = binder.get(self.x)
            code.field(typ, alias, name)
        code.field("int", "hidden", self.shape_template_dict["hidden"])
        code.field("int", "features", self.shape_template_dict["features"])

    def per_sample_code(self, code, binder):
        code.begin_for("int i=0; i<hidden; i++")  # initial cell
        code.assignment("cell[i]", 0)
        code.end()
        code.field("int", "max_length", val="x[0][0].length")
        code.begin_for("int t=0; t<max_length; t++")  # recurrent
        self.per_step_code(code, binder)
        code.end()
        code.begin("int i=0; i<hidden; i++")  # copy to 'last'
        code.assignment("%s[s][i]" % binder.get_name(self.last), "h[s][max_length-1][i]")
        code.end()

    # sample index = "i", step index = "t"
    def per_step_code(self, code, binder):
        # mask = 1
        code.begin_if("mask[s][t] > 0")
        self.union_code(code)  # copy x and state into buffer
        self.mut_code(binder.get_name(self.W_i), "buf", binder.get_name(self.b_i), "input_gate", code)  # gate computing
        self.mut_code(binder.get_name(self.W_o), "buf", binder.get_name(self.b_o), "output_gate", code)
        self.mut_code(binder.get_name(self.W_f), "buf", binder.get_name(self.b_f), "forget_gate", code)
        self.cell_code(code, binder)  # new cell computing
        code.end()
        # mask = 0
        code.begin_else()
        code.begin_if("t > 0")
        code.begin_for("int i=0; i<hidden; i++")  # copy previous hidden state
        code.assignment("h[s][t][i]", "h[s][t-1][i]")
        code.end()
        code.end()
        code.end()

    def union_code(self, code):
        code.begin_for("int k=0; k<hidden; k++")
        code.begin_if("t == 0")
        code.assignment("buf[k]", 0)
        code.end()
        code.begin_else()
        code.assignment("buf[k]", "h[s][t-1][k]")
        code.end()
        code.end()
        code.begin_for("int k=hidden; k<hidden+features; k++")
        code.assignment("buf[k]", "x[s][k-hidden][t]")
        code.end()

    def mut_code(self, W, x, b, target, code, tanh=False):
        code.begin_for("int j=0; j<%s.length; j++" % target)
        code.assignment("%s[j]" % target, 0)
        code.begin_for("int i=0; i<%s.length; i++" % x)
        code.assignment("%s[j]" % target, "%s[i] * %s[i][j]" % (x, W), operator="+=")
        code.end()
        if b:
            code.assignment("%s[j]" % target, b, operator="+=")
        code.end()
        code.begin_for("int j=0; j<%s.length; j++" % target)
        if tanh:
            code.field("double", "exp2_tmp_0", val="Math.exp(2 * %s[j])" % target)
            code.assignment("%s[j]" % target, "(exp2_tmp_0 - 1) / (exp2_tmp_0 + 1)")
        else:
            code.assignment("%s[j]" % target, "1 / (1 + Math.exp(-%s[j]))" % target)
        code.end()

    def cell_code(self, code, binder):
        self.mut_code(binder.get_name(self.W_c), "buf", binder.get_name(self.b_c), "h[s][t]", tanh=True)
        code.begin_for("int i=0; i<hidden; i++")
        code.assignment("cell[i]", "input_gate[i] * h[s][t][i] + forget_gate[i] * cell[i]")
        code.field("double", "exp2_tmp_1", val="Math.exp(2 * cell[i])")
        code.assignment("h[s][t][i]", "output_gate[i] * (exp2_tmp_1 - 1) / (exp2_tmp_1 + 1)")
        code.end()
