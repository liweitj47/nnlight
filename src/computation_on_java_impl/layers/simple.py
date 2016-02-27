from layer.basic.simple import SimpleLayer as SimpleLayerBase


class SimpleLayer(SimpleLayerBase):

    def get_computation_on_java_code(self, code, binder):
        x = binder.get_name(self.x)
        y = binder.get_name(self.y)
        W = binder.get_name(self.W)
        b = binder.get_name(self.b) if self.b else None
        code.field("int", "samples", val="%s.length" % x)
        code.field("int", "features", val="%s[0].length" % x)
        code.field("int", "outputs", val="%s[0].length" % y)
        code.begin_for("int i=0; i<samples; i++")
        code.begin_for("int j=0; j<outputs; j++")
        t = "%s[i][j]" % y
        code.assignment(t, 0)
        code.begin_for("int k=0; k<features; k++")
        code.assignment(t, "%s[i][k] * %s[k][j]" % (x, W), operator="+=")
        code.end()
        if b:
            code.assignment(t, b, operator="+=")
        code.assignment(t, "1 / ( 1 + Math.exp(-%s) )" % t)
        code.end()
        code.end()
