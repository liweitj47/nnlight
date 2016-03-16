from layer.basic.softmax import SoftmaxLayer as SoftmaxLayerBase
from layer.basic.softmax import SequencialSoftmaxLayer as SequencialSoftmaxLayerBase


class SoftmaxLayer(SoftmaxLayerBase):

    def get_computation_on_java_code(self, code, binder):
        datatype = binder.get_base_type(self.x)
        x = binder.get_name(self.x)
        W = binder.get_name(self.W)
        y = binder.get_name(self.y)
        code.field("int", "samples", val="%s.length" % x)
        code.field("int", "features", val="%s[0].length" % x)
        code.field("int", "labels", val="%s[0].length" % W)
        code.begin_for("int i=0; i<samples; i++")
        code.field(datatype, "norm", val=0)
        code.begin_for("int j=0; j<labels; j++")
        code.assignment("%s[i][j]" % y, 0)
        code.begin_for("int k=0; k<features; k++")
        code.assignment("%s[i][j]" % y, "%s[i][k] * %s[k][j]" % (x, W), operator="+=")
        code.end()
        code.assignment("%s[i][j]" % y, "Math.exp(%s[i][j])" % y)
        code.assignment("norm", "%s[i][j]" % y, operator="+=")
        code.end()
        code.begin_for("int j=0; j<labels; j++")
        code.assignment("%s[i][j]" % y, "norm", operator="/=")
        code.end()
        code.end()


class SequencialSoftmaxLayer(SequencialSoftmaxLayerBase):

    def info(self):
        return [
            ("x", "input", ["samples", "length", "features"]),
            ("W", "weight", ["features", "labels"]),
            ("y", "output", ["samples", "length", "labels"])
        ]

    def get_computation_on_java_code(self, code, binder):
        datatype = binder.get_base_type(self.x)
        x = binder.get_name(self.x)
        W = binder.get_name(self.W)
        y = binder.get_name(self.y)
        code.field("int", "samples", val="%s.length" % x)
        code.field("int", "length", val="%s[0].length" % x)
        code.field("int", "features", val="%s[0][0].length" % x)
        code.field("int", "labels", val="%s[0].length" % W)
        code.begin_for("int i=0; i<samples; i++")
        code.begin_for("int w=0; w<length; w++")
        code.field(datatype, "norm", val=0)
        code.begin_for("int j=0; j<labels; j++")
        code.assignment("%s[i][w][j]" % y, 0)
        code.begin_for("int k=0; k<features; k++")
        code.assignment("%s[i][w][j]" % y, "%s[i][w][k] * %s[k][j]" % (x, W), operator="+=")
        code.end()
        code.assignment("%s[i][w][j]" % y, "Math.exp(%s[i][w][j])" % y)
        code.assignment("norm", "%s[i][w][j]" % y, operator="+=")
        code.end()
        code.begin_for("int j=0; j<labels; j++")
        code.assignment("%s[i][w][j]" % y, "norm", operator="/=")
        code.end()
        code.end()
        code.end()
