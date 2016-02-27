from loss.basic.cross_entropy import CrossEntropyLoss as CrossEntropyLossBase
from loss.basic.cross_entropy import SequentialCrossEntropyLoss as SequentialCrossEntropyLossBase
from loss.basic.cross_entropy import BinaryCrossEntropyLoss as BinaryCrossEntropyLossBase
from loss.basic.cross_entropy import SequentialBinaryCrossEntropyLoss as SequentialBinaryCrossEntropyLossBase


class BinaryCrossEntropyLoss(BinaryCrossEntropyLossBase):

    def get_computation_on_java_code(self, code, binder):
        datatype = binder.get_base_type(self.predict)
        x = binder.get_name(self.predict)
        y = binder.get_name(self.golden)
        code.field("int", "samples", val="%s.length" % x)
        code.field(datatype, "sum", val=0)
        code.begin_for("int i=0; i<samples; i++")
        code.assignment("sum",
                        "(%s[i] < 0.00001) ? 0 : %s[i] * Math.log(%s[i])" % (x, y, x),
                        operator="+=")
        code.assignment("sum",
                        "((1-%s[i]) < 0.00001) ? 0 : (1-%s[i]) * Math.log((1-%s[i]))" % (x, y, x),
                        operator="+=")
        code.end()
        code.assignment(binder.get_name("loss"), "sum / samples")


class SequentialBinaryCrossEntropyLoss(SequentialBinaryCrossEntropyLossBase):

    def get_computation_on_java_code(self, code, binder):
        datatype = binder.get_base_type(self.predict)
        x = binder.get_name(self.predict)
        y = binder.get_name(self.golden)
        code.field("int", "samples", val="%s.length" % x)
        code.field("int", "length", val="%s[0].length" % x)
        code.field(datatype, "sum", val=0)
        code.begin_for("int i=0; i<samples; i++")
        code.begin_for("int j=0; i<length; j++")
        code.assignment("sum",
                        "(%s[i][j] < 0.00001) ? 0 : %s[i][j] * Math.log(%s[i][j])" % (x, y, x),
                        operator="+=")
        code.assignment("sum",
                        "((1-%s[i][j]) < 0.00001) ? 0 : (1-%s[i][j]) * Math.log((1-%s[i][j]))" % (x, y, x),
                        operator="+=")
        code.end()
        code.end()
        code.assignment(binder.get_name("loss"), "sum / (samples * length)")


class CrossEntropyLoss(CrossEntropyLossBase):

    def get_computation_on_java_code(self, code, binder):
        datatype = binder.get_base_type(self.predict)
        x = binder.get_name(self.predict)
        y = binder.get_name(self.golden)
        code.field("int", "samples", val="%s.length" % x)
        code.field("int", "labels", val="%s[0].length" % x)
        code.field(datatype, "sum", val=0)
        code.begin_for("int i=0; i<samples; i++")
        code.begin_for("int j=0; j<labels; j++")
        code.assignment("sum",
                        "(%s[i][j] < 0.00001) ? 0 : %s[i][j] * Math.log(%s[i][j])" % (x, y, x),
                        operator="+=")
        code.end()
        code.end()
        code.assignment(binder.get_name(self.loss), "-sum / samples")


class SequentialCrossEntropyLoss(SequentialCrossEntropyLossBase):

    def get_computation_on_java_code(self, code, binder):
        datatype = binder.get_base_type(self.predict)
        x = binder.get_name(self.predict)
        y = binder.get_name(self.golden)
        code.field("int", "samples", val="%s.length" % x)
        code.field("int", "length", val="%s[0].length" % x)
        code.field("int", "labels", val="%s[0][0].length" % x)

        code.field(datatype, "sum", val=0)
        code.begin_for("int i=0; i<samples; i++")
        code.begin_for("int w=0; w<length; w++")
        code.begin_for("int j=0; j<labels; j++")
        code.assignment("sum",
                        "(%s[i][w][j] < 0.00001) ? 0: %s[i][w][j] * Math.log(%s[i][w][j])" % (x, y, x),
                        operator="+=")
        code.end()
        code.end()
        code.end()
        code.assignment(binder.get_name("loss"), "-sum / (samples * length)")
