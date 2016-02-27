from layer.basic.logistic import LogisticLayer as LogisticLayerBase


class LogisticLayer(LogisticLayerBase):

    def get_computation_on_java_code(self, code, binder):
        x = binder.get_name(self.x)
        y = binder.get_name(self.y)
        W = binder.get_name(self.W)
        b = binder.get_name(self.b) if self.b else None
        code.field("int", "samples", val="%s.length" % x)
        code.field("int", "features", val="%s[0].length" % x)
        code.begin_for("int i=0; i<samples; i++")
        t = "%s[i]" % y
        code.assignment(t, 0)
        code.begin_for("int j=0; j<features; j++")
        code.assignment(t, "%s[i][j] * %s[j]" % (x, W), operator="+=")
        code.end()
        if b:
            code.assignment(t, b, operator="+=")
        code.assignment(t, "1 / ( 1 + Math.exp(-%s) )" % t)
        code.end()
