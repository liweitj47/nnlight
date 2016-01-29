import theano
from theano import tensor
from layers import Layer, LayerWithData
from utils.constructor import Constructor


class InputLayer(LayerWithData):

    def __init__(self, name, shape, dtype, data):
        Layer.__init__(self)
        self.name = name
        self.data = data
        constructor = Constructor.get_default_array_constructor(dtype)
        if not constructor:
            self.error("invalid datatype '%s' for input layer '%s'" % (dtype, name))
        self.output = constructor(shape, self)

    def get_inputs(self):
        return []

    def get_output(self, name=None):
        if name is not None:
            return None
        else:
            return self.output

    def get_outputs(self):
        return [self.get_output()]

    def get_shape(self):
        return self.get_output().get_shape()

    def get_data(self):
        return self.data

    def set_data(self, data):
        self.data = data

    def forward_shape(self, override=False):
        shape0 = self.data.shape
        shape1 = self.get_output().get_shape()
        if len(shape0) != len(shape1):
            self.error("inconsistent input data dimension for '%s',%d expected "
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
        self.get_output().set_shape(shape1)

    def backward_shape(self, override=False):
        shape0 = self.data.shape
        shape1 = self.get_output().get_shape()
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
            block = self.get_data()[0:maximum_sample_size]
            setattr(self, "theano_shared_data", theano.shared(block, borrow=True))
        return getattr(self, "theano_shared_data")

    def set_theano_shared(self, data):
        shared = getattr(self, "theano_shared_data")
        shared.set_value(data)
