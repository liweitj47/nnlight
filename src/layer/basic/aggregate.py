from theano import tensor
from value.values import NNValue
from layer.layers import Layer


class AggregateLayer(Layer):

    def __init__(self, name, params, core):
        Layer.__init__(self, name, params, core)
        inputs = params.get("inputs")
        if inputs is None:
            self.error("missing 'inputs' field for layer '%s'" % self.name)
        if isinstance(inputs, NNValue):
            inputs = [inputs]
        self.check(isinstance(inputs, list) and len(inputs) > 0 and all([isinstance(_, NNValue) for _ in inputs]),
                   "the 'inputs' field of layer '%s' should be a list of NNValues" % self.name)

        dtype, shape = inputs[0].get_dtype(), inputs[0].get_shape()
        from utility.constructor import Constructor
        self.output = Constructor.create_value(self, len(shape), dtype)
        self.inputs = inputs
        self.axis = params.get("axis", -1)
        self.check(isinstance(self.axis, int) and self.axis < len(self.output.get_shape()),
                   "parameter 'axis' should be an integer < dimension of the input")

    def get_inputs(self):
        return self.inputs

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
        input_shapes = [_.get_shape() for _ in self.get_inputs()]
        output_shape = self.get_outputs()[0].get_shape()
        template_shape = [-1 for _ in input_shapes[0]]
        for i, shape in enumerate(input_shapes):
            if len(shape) != len(template_shape):
                self.error("inconsistent dimension for '%s''s %dth "
                           "input, expected to be %d but actually %d"
                           % (self.name, i, len(template_shape), len(shape)))
            for j, val in enumerate(shape):
                if j == self.axis:
                    continue
                if val <= 0:
                    if template_shape[j] > 0:
                        input_shapes[j] = val
                        output_shape[j] = val
                else:
                    output_shape[j] = val
                    if template_shape[j] <= 0:
                        template_shape[j] = val
                    elif val != template_shape[j]:
                        self.error("inconsistent shape value for '%s''s %dth input,"
                                   "%dth element expected to be %d, but actually %d"
                                   % (self.name, i, j, template_shape[j], val))
        self.get_outputs()[0].set_shape(output_shape)
        for i, inp in enumerate(self.get_inputs()):
            inp.set_shape(input_shapes[i])

    def backward_shape(self, override=False):
        input_shapes = [_.get_shape() for _ in self.get_inputs()]
        output_shape = self.get_outputs()[0].get_shape()
        for i, shape in enumerate(input_shapes):
            for j, val in enumerate(output_shape):
                if j == self.axis:
                    continue
                if val > 0:
                    if shape[j] <= 0:
                        shape[j] = val
                    elif val != shape[j]:
                        self.error("inconsistent shape value for '%s''s %dth input,"
                                   "%dth element expected to %d, but actually %d"
                                   % (self.name, i, j, val, shape[j]))
        for i, inp in enumerate(self.get_inputs()):
            inp.set_shape(input_shapes[i])


class ConcatLayer(AggregateLayer):

    def __init__(self, name, params, core):
        AggregateLayer.__init__(self, name, params, core)
        if self.axis < 0:
            self.axis = 1

    def forward_shape(self, override=False):
        AggregateLayer.forward_shape(self, override)
        input_shapes = [_.get_shape() for _ in self.get_inputs()]
        output_shape = self.get_outputs()[0].get_shape()

        if all([shape[self.axis] > 0 for shape in input_shapes]):
            aggregate_dim = sum([shape[self.axis] for shape in input_shapes])
            if output_shape[self.axis] > 0:
                if aggregate_dim != output_shape[self.axis]:
                    if override:
                        output_shape[self.axis] = aggregate_dim
                    else:
                        self.error("inconsistent shape value for '%s''s output,"
                                   "%dth element expected to be %d, but actually %d"
                                   % (self.name, self.axis, aggregate_dim, output_shape[self.axis]))
            else:
                output_shape[self.axis] = aggregate_dim
            self.get_outputs()[0].set_shape(output_shape)

    def get_theano_output(self, diagram):
        inputs = [diagram.get(inp) for inp in self.get_inputs()]
        return tensor.concatenate(inputs, axis=self.axis)
