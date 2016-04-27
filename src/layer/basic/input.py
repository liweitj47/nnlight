import theano
from theano import tensor
from utility.typecheck import TypeChecker
from layer.layers import Layer, LayerWithData
from layer.dropoutable import Dropoutable


class InputLayer(LayerWithData, Dropoutable):

    def __init__(self, name, params, core):
        Layer.__init__(self, name, params, core)
        Dropoutable.__init__(self, params.get("dropout", 0.))
        dtype = params["dtype"]
        shape = params["shape"]
        self.data = params["data"]
        from utility.constructor import Constructor
        constructor = Constructor.get_default_array_constructor(dtype)
        if not constructor:
            self.error("invalid datatype '%s' for input layer '%s'" % (dtype, name))
        self.output = constructor(shape, self)

    def get_inputs(self):
        return []

    def get_value(self, name=None):
        if name is not None:
            return None
        else:
            return self.output

    def get_outputs(self):
        return [self.get_value()]

    def get_shape(self):
        return self.get_value().get_shape()

    def get_data(self):
        return self.data

    def set_data(self, data):
        self.data = data

    def check_input_type(self):
        if self.data is not None:
            self.check(TypeChecker.consistent(self.data.dtype, self.output.get_dtype()),
                       "inconsistent datatype of input '%s', '%s' expected "
                       "but actually '%s'" % (self.name, self.output.get_dtype(), self.data.dtype))

    def forward_shape(self, override=False):
        shape0 = [int(x) for x in self.data.shape]  # ensure integers be int type
        shape1 = self.get_value().get_shape()
        if len(shape0) != len(shape1):
            self.error("inconsistent input data dimension for '%s', %d expected "
                       "but actually %d" % (self.name, len(shape1), len(shape0)))
        for i in range(len(shape0)):
            if shape0[i] == shape1[i]:
                continue
            elif shape1[i] <= 0 or override:
                shape1[i] = shape0[i]
            else:
                expected = [d if d > 0 else 'x' for d in shape1]
                self.error("inconsistent input data shape for '%s', %s expected "
                           "but actually %s" % (self.name, expected, shape0))
        self.get_value().set_shape(shape1)

    def backward_shape(self, override=False):
        shape0 = [int(x) for x in self.data.shape]  # ensure integers be int type
        shape1 = self.get_value().get_shape()
        if len(shape0) != len(shape1):
            self.error("inconsistent input data dimension for '%s', %d expected "
                       "but actually %d" % (self.name, len(shape1), len(shape0)))
        for i in range(len(shape0)):
            if shape0[i] != shape1[i] and shape1[i] > 0:
                expected = [d if d > 0 else 'x' for d in shape1]
                self.error("inconsistent input data shape for '%s', %s expected "
                           "but actually %s" % (self.name, expected, shape0))

    #  theano functions
    def get_theano_output(self, diagram):
        dims = len(self.get_shape())
        if dims == 1:
            return theano.tensor.vector()
        elif dims == 2:
            return theano.tensor.matrix()
        elif dims == 3:
            return theano.tensor.tensor3()
        elif dims == 4:
            return theano.tensor.tensor4()
        else:
            self.error("dimension %d not supported for input data '%s'" % (dims, self.name))

    def get_theano_shared(self, maximum_sample_size):
        if not hasattr(self, "theano_shared_data"):
            if not maximum_sample_size:
                block = self.get_data()
            else:
                block = self.get_data()[0:maximum_sample_size]
            setattr(self, "theano_shared_data", theano.shared(block, borrow=True))
        return getattr(self, "theano_shared_data")

    def set_theano_shared(self, data):
        shared = getattr(self, "theano_shared_data")
        shared.set_value(data)
