import random
import numpy
import theano
from utility.typecheck import TypeChecker
from layer.layers import Layer, LayerWithData
from theano import tensor


class WeightLayer(LayerWithData):

    def __init__(self, name, params, core):
        Layer.__init__(self, name, params, core)
        self.data = None
        self.init_method = params["init_method"]
        shape = params["shape"]
        dtype = params["dtype"]
        from utility.constructor import Constructor
        if len(shape) > 0:
            constructor = Constructor.get_default_array_constructor(dtype)
            if not constructor:
                self.error("invalid datatype '%s' for weight layer '%s'" % (dtype, name))
            self.output = constructor(shape, self)
        else:
            constructor = Constructor.get_default_constructor(dtype)
            if not constructor:
                self.error("invalid datatype '%s' for weight layer '%s'" % (dtype, name))
            self.output = constructor(self)

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
        # note: use its output value shape, which can be inferred before form real data
        return self.get_value().get_shape()

    def get_data(self):
        if hasattr(self, "theano_shared_data"):
            return getattr(self, "theano_shared_data").get_value()
        else:
            return self.data

    def set_data(self, data):
        self.data = data

    def check_input_type(self):
        if self.data is not None:
            self.check(TypeChecker.consistent(self.data.dtype, self.output.get_dtype()),
                       "inconsistent datatype of weight '%s', '%s' expected "
                       "but actually '%s'" % (self.name, self.output.get_dtype(), self.data.dtype))

    def forward_shape(self, override=False):
        if self.data is None:
            return  # no more information to forward, this is not the case when input layer is reset with new data
        shape0 = [int(x) for x in self.data.shape]  # ensure integers be int type
        shape1 = self.get_value().get_shape()
        if len(shape0) != len(shape1):
            self.error("inconsistent weight dimension for '%s', %d expected "
                       "but actually %d" % (self.name, len(shape1), len(shape0)))
        for i in range(len(shape0)):
            if shape0[i] == shape1[i]:
                continue
            elif shape1[i] <= 0 or override:
                shape1[i] = shape0[i]
            else:
                expected = [d if d > 0 else 'x' for d in shape1]
                self.error("inconsistent weight shape for '%s', %s expected "
                           "but actually %s" % (self.name, expected, shape0))
        self.get_value().set_shape(shape1)

    def backward_shape(self, override=False):
        if self.data is None:
            return  # no more information to backward
        shape0 = [int(x) for x in self.data.shape]  # ensure integers be int type
        shape1 = self.get_value().get_shape()
        if len(shape0) != len(shape1):
            self.error("inconsistent weight dimension for '%s', %d expected "
                       "but actually %d" % (self.name, len(shape1), len(shape0)))
        for i in range(len(shape0)):
            if shape0[i] != shape1[i] and shape1[i] > 0:
                expected = [d if d > 0 else 'x' for d in shape1]
                self.error("inconsistent weight shape for '%s', %s expected "
                           "but actually %s" % (self.name, expected, shape0))

    #  theano functions
    def get_theano_output(self, diagram):
        dims = len(self.get_shape())
        if dims == 0:
            return theano.tensor.scalar()
        elif dims == 1:
            return theano.tensor.vector()
        elif dims == 2:
            return theano.tensor.matrix()
        elif dims == 3:
            return theano.tensor.tensor3()
        elif dims == 4:
            return theano.tensor.tensor4()
        else:
            self.error("dimension %d not supported for weight '%s'" % (dims, self.name))

    def get_theano_shared(self, maximum_sample_size):
        if self.data is None:
            self.data = numpy.zeros(self.get_shape(), dtype="float32")
            if self.init_method:
                high = 4.0 * numpy.sqrt(6.0 / sum(self.get_shape()))
                low = -high

                def __randomize__(w, dims):
                    if len(dims) == 0:
                        w += numpy.asarray(random.uniform(low, high), dtype="float32")
                    elif len(dims) == 1:
                        for i in range(dims[0]):
                            w[i] = random.uniform(low, high)
                    else:
                        for i in range(dims[0]):
                            __randomize__(w[i], dims[1:])
                __randomize__(self.data, self.get_shape())

        if not hasattr(self, "theano_shared_data"):
            shared = theano.shared(self.data, borrow=True)
            setattr(self, "theano_shared_data", shared)
        else:
            getattr(self, "theano_shared_data").set_value(self.data)
        return getattr(self, "theano_shared_data")

    def set_theano_shared(self, data):
        shared = getattr(self, "theano_shared_data")
        shared.set_value(data)
