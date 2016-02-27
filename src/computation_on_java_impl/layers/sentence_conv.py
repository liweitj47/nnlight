from layer.basic.sentence_conv import SentenceConvolutionLayer as SentenceConvolutionLayerBase


class SentenceConvolutionLayer(SentenceConvolutionLayerBase):

    """
    notation for shapes
    ("sentence", "input", ["samples", "features", "max_length"]),
    ("mask", "input", ["samples", "max_length"]),
    ("filter", "weight", ["depth", "features", "window"]),
    ("output", "output", ["samples", "max_length - window + 1", "depth"])
    """
    def get_computation_on_java_code(self, code, binder):
        code.field("int", "samples", val=binder.get_name(self.sentence) + ".length")
        code.field("int", "features", val=binder.get_name(self.sentence) + "[0].length")
        code.field("int", "length", val=binder.get_name(self.sentence) + "[0][0].length")
        code.field("int", "window", val=binder.get_name(self.filter) + "[0][0].length")
        code.field("int", "depth", val=binder.get_name(self.filter) + ".length")
        code.begin_for("int i=0; i<samples; i++")
        code.begin_for("int w=0; w<length-window+1; w++")
        code.begin_for("int d=0; d<depth; d++")
        target = binder.get_name(self.output) + "[i][w][d]"
        code.assignment(target, 0)
        code.begin_if(binder.get_name(self.mask) + "[i][w+window-1] != 0")
        code.begin_for("int j=0; j<features; j++")
        code.begin_for("int k=0; k<window; k++")
        val = "%s[i][j][w+k] * %s[d][features-j-1][window-k-1]" % \
              (binder.get_name(self.sentence), binder.get_name(self.filter))
        code.assignment(target, val, operator="+=")
        code.end()
        code.end()
        code.end()
        code.end()
        code.end()
        code.end()
