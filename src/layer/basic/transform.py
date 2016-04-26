from value.values import NNValue
from layer.layers import Layer


class DimShuffle(Layer):

    def __init__(self, name, params, core):
        Layer.__init__(self, name, params, core)
        x = params.get("x")
        if x is None:
            self.error("missing input field 'x' for layer '%s'" % self.name)
        self.check(isinstance(x, NNValue),
                   "the input field 'x' of layer '%s' should be a NNValue" % self.name)
        dims = params.get("dims")
        if dims is None:
            self.error("missing 'dims' field for layer '%s'" % self.name)
        self.check(isinstance(dims, list) and len(dims) > 0 and
                   all([isinstance(_, int) or _ == "x" for _ in dims]),
                   "the 'dims' field of layer '%s' should be a list of integers" % self.name)
        from utility.constructor import Constructor
        self.output = Constructor.create_value(self, len(dims), x.get_dtype())
        self.input = x
        self.dims = dims

    def get_inputs(self):
        return [self.input]

    def get_outputs(self):
        return [self.output]

    def get_value(self, name=None):
        if name is not None:
            return None
        else:
            return self.output

    def check_input_type(self):
        pass

    def forward_shape(self, override=False):
        input_shape = self.input.get_shape()
        output_shape = self.output.get_shape()
        for i, src in enumerate(self.dims):
            if isinstance(src, int):
                self.check(0 <= src < len(input_shape),
                           "index out of bound for 'dims' field's %dth "
                           "element: %d" % (i, src))
                d_i = input_shape[src]
                d_o = output_shape[i]
                if d_i > 0:
                    if d_i != d_o > 0 and not override:
                        self.error("inconsistent shape value for '%s''s output,"
                                   "%dth element expected to be %d, but actually %d"
                                   % (self.name, i, d_i, d_o))
                    else:
                        output_shape[i] = d_i
        self.output.set_shape(output_shape)

    def backward_shape(self, override=False):
        input_shape = self.input.get_shape()
        output_shape = self.output.get_shape()
        for i, src in enumerate(self.dims):
            if isinstance(src, int):
                self.check(0 <= src < len(input_shape),
                           "index out of bound for 'dims' field's %dth "
                           "element: %d" % (i, src))
                d_i = input_shape[src]
                d_o = output_shape[i]
                if d_o > 0:
                    if d_o != d_i > 0:
                        self.error("inconsistent shape value for '%s''s input,"
                                   "%dth element expected to be %d, but actually %d"
                                   % (self.name, i, d_o, d_i))
                    elif d_i <= 0:
                        input_shape[src] = d_o
        self.input.set_shape(input_shape)

    def get_theano_output(self, diagram):
        return diagram.get(self.input).dimshuffle(self.dims)
