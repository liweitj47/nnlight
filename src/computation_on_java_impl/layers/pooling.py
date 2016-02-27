from layer.basic.pooling import MaxPoolingWithTimeLayer as MaxPoolingWithTimeLayerBase


class MaxPoolingWithTimeLayer(MaxPoolingWithTimeLayerBase):

    def get_computation_on_java_code(self, code, binder):
        datatype = binder.get_base_type(self.input)
        input_var = binder.get_name(self.input)
        output_var = binder.get_name(self.output)
        code.field("int", "samples", val=input_var + ".length")
        code.field("int", "length", val=input_var + "[0].length")
        code.field("int", "features", val=input_var + "[0][0].length")
        code.begin_for("int i=0; i<samples; i++")
        code.begin_for("int j=0; j<features; j++")
        code.field(datatype, "maximum", val=input_var + "[i][0][j]")
        code.begin_for("int k=1; k<length; k++")
        code.begin_if("maximum < %s[i][k][j]" % input_var)
        code.assignment("maximum", "%s[i][k][j]" % input_var)
        code.end()
        code.end()
        code.assignment(output_var + "[i][j]", "maximum")
        code.end()
        code.end()
