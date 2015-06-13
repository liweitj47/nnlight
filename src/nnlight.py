# -*- coding: utf-8 -*-
"""

"""
import random
import string
import theano
import numpy


class NNDebug:
    """
    debug utility class
    """
    def __init__(self):
        pass

    @staticmethod
    def error_(msg):
        raise Exception(msg)

    @staticmethod
    def assert_(cond, msg):
        if not cond:
            raise Exception(msg)


global_config = {}


def get_default_config():

    # NNValue constructor mapping
    cm = {
        NNScalarInt64: ["int64", "int"],
        NNScalarFloat32: ["float32", "float"],
        NNArrayFloat32: ["[float32]", "[float]"],
        NNArrayInt64: ["[int64]", "[int]"],

        SentenceConvLayer: ["sentence_convolution"],
        MaxPoolingLayer: ["max_pooling"],
        SoftmaxLayer: ["softmax"],
        LogisticLayer: ["logistic"],

        BinaryCrossEntropyLoss: ["binary_cross_entropy"],
        CrossEntropyLoss: ["cross_entropy"]
    }

    constructor_map = {}
    for constructor, datatypes in cm.items():
        for datatype in datatypes:
            constructor_map[datatype] = constructor
    global_config["constructor_map"] = constructor_map

    # NNUpdate constructor mapping
    update_cm = {
        SGDUpdate: ["sgd"]
    }

    constructor_map = {}
    for constructor, datatypes in update_cm.items():
        for datatype in datatypes:
            constructor_map[datatype] = constructor
    global_config["update_constructor_map"] = constructor_map


def create(src, data_dict=None):
    """
    create the network object from configuration source and
    assign data to the network

    :param src:
        if str, the configuration file path
        if dict, configuration python dictionary

    :param data_dict:
        dict, the key is the network input name defined by
            the configurations, and the value is the
            numpy.ndarray fed to the network's input

    :return: the NNBase network object built
    """
    if len(global_config) == 0:
        get_default_config()
    if data_dict is None:
        data_dict = {}
    if isinstance(src, str):
        return create_from_file(src, data_dict)
    else:
        return create_from_dict(src, data_dict)


def create_from_file(config_path, data_dict=None):
    """
    create the network object from configuration file and
    assign data to the network

    :param config_path:
        str, the configuration file path

    :param data_dict:
        dict, the key is the network input name defined by
            the configurations, and the value is the
            numpy.ndarray fed to the network's input

    :return: the NNBase network object built
    """
    try:
        config_file = open(config_path)
    except IOError:
        NNDebug.error_("unable to open configuration file '%s" % config_path)
        return None

    def remove_quote(line):
        return line[:line.find("#")].strip()

    lines = map(remove_quote, config_file.readlines())
    config_file.close()

    idx = 0
    length = len(lines)

    list_sections = {"paramlist": "param"}
    single_sections = ["training"]

    d = {
        "param": [],
        "input": [],
        "weight": [],
        "layer": [],
        "loss": [],
        "training": None
    }

    def eval_(v):
        v = v.strip()
        if v.startswith("[") and v.endswith("]"):
            return map(eval_,
                       v[1:-1].split(","))
        elif v.find(",") >= 0:
            return map(eval_,
                       filter(lambda _: _.strip() != "",
                              v.split(",")))
        elif v.startswith("{"):
            return dict([
                (key, eval_(value))
                for key, value in
                filter(lambda _: len(_) == 2,
                       [_.split(":") for _ in v[1:-1].split(",")])
            ])
        else:
            try:
                v = string.atoi(v)
            except ValueError:
                try:
                    v = string.atof(v)
                except ValueError:
                    pass
            finally:
                return v

    def read_block_(i):
        while i < length:
            line = lines[i]
            if line.startswith("[") and \
                    line.endswith("]"):
                break
            i += 1
        else:
            return i

        item = {}
        section = lines[i][1:-1].strip()
        i += 1

        while i < length:
            line = lines[i]
            if line.startswith("["):
                break
            i += 1
            eq_idx = line.find("=")
            if eq_idx < 0:
                continue
            key = line[:eq_idx].strip()
            value = line[eq_idx+1:].strip()
            item[key] = eval_(value)

        if section in list_sections:
            d[list_sections[section]] += \
                [{"name": k, "value": v}
                 for k, v in item.items()]
        elif section in single_sections:
            d[section] = item
        elif section in d:
            d[section].append(item)

        return i

    while idx < length:
        idx = read_block_(idx)

    return create_from_dict(d, data_dict)


def create_from_dict(d, data_dict=None):
    """
    create the network object from configuration dictionary and
    assign data to the network

    :param d:
        dict, the configuration dictionary

    :param data_dict:
        dict, the key is the network input name defined by
            the configurations, and the value is the
            numpy.ndarray fed to the network's input

    :return: the NNBase network object built
    """

    # basic objects
    if data_dict is None:
        data_dict = {}
    nn = NNBase()
    core = nn.get_core()
    config = global_config

    # debug utilities
    def error_(s):
        NNDebug.error_("[NN Constructor] " + str(s))

    def assert_(cond, s):
        NNDebug.assert_(cond, "[NN Constructor] " + str(s))

    # build NNValue from non-uniform values
    def eval_(v):
        if isinstance(v, str):
            refs = v.strip().split(".")
            prefix = []
            node = core["values"].get(refs[0])
            if not node:
                error_("value '%s' undefined" % refs[0])
            prefix.append(refs[0])
            node = node.node()

            for ref in refs[1:]:
                node = node.node(ref=ref)
                if not node:
                    error_("value '%s' undefined in '%s'<%s>" % (ref, str.join(".", prefix), node))
                prefix.append(ref)

            return node

        elif isinstance(v, int):
            return NNScalarInt64(v)

        elif isinstance(v, float):
            return NNScalarFloat32(v)

        elif isinstance(v, NNValue):
            return v

        else:
            error_("unknown value '%s' for eval_()" % v)

    # extract the wrapped value from input
    def raw_(v):
        if isinstance(v, NNValue):
            return v.value()
        elif isinstance(v, list):
            return map(raw_, v)
        else:
            error_("unknown value '%s' for raw_()" % v)

    # build input NNValue object
    def make_input_(shape, datatype):
        if isinstance(shape, int):
            shape = [None for _ in range(shape)]

        assert_(isinstance(shape, list), "the shape of input should be list")
        assert_(isinstance(datatype, str), "the datatype of input should be str")

        shape = [None if x is None else raw_(eval_(x))
                 for x in shape]

        if len(shape) == 0:
            constructor = config["constructor_map"].get(datatype)
            if constructor:
                return constructor()
        else:
            datatype = "[" + datatype + "]"
            constructor = config["constructor_map"].get(datatype)
            if constructor:
                return constructor(shape=shape, v=None)
        error_("unknown datatype '%s' for input" % datatype)

    # build shared NNValue object
    def make_shared_(shape, init_method, datatype):

        assert_(isinstance(datatype, str),
                "the datatype of weight should be str")
        assert_(datatype in ["float32", "int64"],
                "datatype '%s' not supported for weights" % datatype)
        assert_(datatype in config["constructor_map"],
                "datatype '%s' not supported for weights" % datatype)
        basetype = datatype
        datatype = "[" + basetype + "]"

        shape = map(eval_, shape)

        assert_((
            isinstance(shape, list) and
            all([isinstance(_, NNScalarInt) for _ in shape]) and
            all([_.has_raw() for _ in shape])
        ), "shape of weight must be an constant integer array")

        shape = map(raw_, shape)

        if len(shape) == 0:
            constructor = config["constructor_map"].get(datatype)
            return constructor(v=0.0, shared=True)

        v = numpy.zeros(shape, dtype=basetype)
        if init_method == "random":
            high = 4.0 * numpy.sqrt(6.0 / sum(shape))
            low = -high

            def __randomize__(w, dims):
                if len(dims) == 1:
                    for i in range(dims[0]):
                        w[i] = random.uniform(low, high)
                else:
                    for i in range(dims[0]):
                        __randomize__(w[i], dims[1:])

            __randomize__(v, shape)

        constructor = config["constructor_map"][datatype]
        return constructor(shape=shape, v=v)

    # build middle node NNValue
    def make_layer_(layer):
        layername = layer.get("name")
        layertype = layer.get("type")
        if not layertype:
            error_("missing type for layer '%s'" % layername)

        constructor = config["constructor_map"].get(layertype)
        if not constructor:
            error_("unknown layer type '%s'" % layertype)

        inputs = layer.get("input", [])
        if not isinstance(inputs, list):
            inputs = [inputs]
        inputs = map(eval_, inputs)
        params = dict([(k, eval_(v)) for k, v in layer.get("param", {}).items()])

        layer_obj = constructor(layername, inputs, params)
        assert_(isinstance(layer_obj, Layer), "datatype for layer should be Layer")
        return layer_obj

    # build loss function node NNValue
    def make_loss_(loss):
        lossname = loss.get("name")
        losstype = loss.get("type")
        if not losstype:
            error_("missing type for loss '%s'" % lossname)

        constructor = config["constructor_map"].get(losstype)
        if not constructor:
            error_("unknown loss type '%s'" % losstype)

        inputs = loss.get("input", [])
        if not isinstance(inputs, list):
            inputs = [inputs]
        inputs = map(eval_, inputs)
        params = dict([(k, eval_(v)) for k, v in loss.get("param", {}).items()])

        loss_obj = constructor(lossname, inputs, params)
        assert_(isinstance(loss_obj, Loss), "datatype for loss should be Loss")
        return loss_obj

    # build NNValue from raw data
    def wrap_data_(data):
        if isinstance(data, float):
            return NNScalarFloat32(v=data, shared=True)
        elif isinstance(data, int):
            return NNScalarInt64(v=data, shared=True)
        elif isinstance(data, numpy.ndarray):
            return NNArrayFloat32(v=data, shape=data.shape)
        else:
            error_("type '%s' not supported for input data" % type(data))

    # build NNUpdate
    def make_update_(loss, ws, params):
        assert_(isinstance(loss, Loss),
                "loss function should be Loss")
        assert_(all([isinstance(w, NNValue) and w.has_raw()
                     for w in ws.values()]),
                "update targets should be durable NNValue")

        gradtype = params.get("method", "sgd")
        constructor = config["update_constructor_map"].get(gradtype)
        if not constructor:
            error_("unknown update type '%s'" % gradtype)

        update_obj = constructor(loss, ws, params)
        assert_(isinstance(update_obj, Update), "datatype for update should be Update")
        return update_obj

    # check name validity
    def check_name_(s):
        if (not isinstance(s, str)) or s == "":
            error_("bad format name '%s'" % s)
        elif s in core["values"]:
            error_("duplicate definition '%s'" % s)

    # start building process

    # parameters
    for item in d.get("param", []):
        name = item.get("name", "")
        check_name_(name)
        value = item.get("value")
        if value is None:
            error_("missing value for parameter '%s'" % name)
        core["values"][name] = eval_(value)

    # inputs
    for item in d.get("input", []):
        name = item.get("name", "")
        check_name_(name)
        shp = item.get("shape")
        if shp is None:
            error_("missing shape for input '%s'" % name)
        dtype = item.get("type", "float32")
        value = make_input_(shp, dtype)
        core["values"][name] = value
        core["inputs"][name] = value
        if name in data_dict:
            core["data"][name] = wrap_data_(data_dict[name])

    # weights
    for item in d.get("weight", []):
        name = item.get("name", "")
        check_name_(name)
        shp = item.get("shape", [])
        init = item.get("init", "random")
        dtype = item.get("type", "float32")
        value = make_shared_(shp, init, dtype)
        core["values"][name] = value
        core["weights"][name] = value
        if item.get("update", True):
            core["learn"][name] = value

    # layers
    for item in d.get("layer", []):
        name = item.get("name", "")
        check_name_(name)
        core["values"][name] = make_layer_(item)

    # losses
    for item in d.get("loss", []):
        name = item.get("name", "")
        check_name_(name)
        value = make_loss_(item)
        core["values"][name] = value

    # training
    item = d.get("training", None)
    if not item:
        error_("missing training section")
    if "loss" not in item:
        error_("missing loss function")
    core["loss"] = eval_(item["loss"])
    core["updates"] = make_update_(
        core["loss"],
        core["learn"],
        {
            "learning_rate": item.get("learning_rate", 0.1),
            "method": item.get("method", "sgd")
        }
    )
    test_info = item.get("test_info", [])
    if not isinstance(test_info, list):
        test_info = [test_info]
    core["test_info"] = [(n, eval_(n)) for n in test_info]

    train_info = item.get("train_info", [])
    if not isinstance(train_info, list):
        train_info = [train_info]
    core["train_info"] = [(n, eval_(n)) for n in train_info]

    return nn


class NNValue(object):
    """
    wrapper class for underlying DAG nodes that
    are operated in actual. All nodes in the
    framework should inherit from this class
    """

    @staticmethod
    def error_(s):
        NNDebug.error_(("[NNValue] ", s))

    @staticmethod
    def assert_(cond, s):
        NNDebug.assert_(cond, s)

    def node(self, ref=None):
        """
        get the referenced NNValue (sub)object

        :param ref:
            if str, is the reference name for sub NNValue to
            be got; if None, return self

        :return: None if ref is not found, else NNValue
        """
        if ref is None:
            return self
        else:
            return None

    def value(self):
        self.error_("value() method not implement for %s" % type(self))

    def update(self, new_value):
        """
        set the new wrapped value, raise exception when
        inconsistent value met

        :param new_value: value to be replaced
        """
        self.error_("update() method not implement for %s" % type(self))

    def raw(self):
        """
        get the durable raw data for the underlying node, raise
        exception when not a durable node

        :return: raw data for underlying node
        """
        self.error_("raw() method not implement for %s" % type(self))

    def has_raw(self):
        """
        :return: boolean, whether has durable raw data
        """
        self.error_("has_raw() method not implement for %s" % type(self))


class NNScalar(NNValue):
    def __init__(self, v=None, shared=False):
        NNValue.__init__(self)
        if v is None:
            self.v = theano.tensor.scalar()
        elif isinstance(v, int) or isinstance(v, float):
            self.v = theano.shared(value=v, borrow=True) if shared else v
        else:
            self.v = v

    def value(self):
        return self.v

    def update(self, new_value):
        self.assert_(hasattr(self.v, "set_value"),
                     "invalid update for non-shared target value")

        self.assert_(isinstance(new_value, int) or
                     isinstance(new_value, float) or
                     (isinstance(new_value, numpy.ndarray) and
                      len(new_value.shape) == 0),
                     "inconsistent new value '%s'" % new_value)

        self.v.set_value(new_value.get_value())

    def raw(self):
        if hasattr(self.v, "get_value"):
            return self.v.get_value()
        elif isinstance(self.v, int) or \
                isinstance(self.v, float):
            return self.v
        else:
            self.error_("only durable NNScalar object has raw value")

    def has_raw(self):
        return hasattr(self.v, "get_value") or \
            isinstance(self.v, int) or \
            isinstance(self.v, float)


class NNScalarInt(NNScalar):
    def __init__(self, v=None, shared=False):
        NNScalar.__init__(self, v, shared)


class NNScalarInt64(NNScalarInt):
    def __init__(self, v=None, shared=False):
        NNScalarInt.__init__(self, v, shared)


class NNScalarFloat32(NNScalar):
    def __init__(self, v=None, shared=False):
        NNScalar.__init__(self, v, shared)


class NNArray(NNValue):
    def __init__(self, shape, v=None):
        NNValue.__init__(self)
        self.shp = shape
        if v is None:
            if len(shape) > 4:
                self.error_("NNArray support at most 4DArray temporarily")
            self.v = [theano.tensor.vector,
                      theano.tensor.matrix,
                      theano.tensor.tensor3,
                      theano.tensor.tensor4][len(shape) - 1]()
        elif isinstance(v, numpy.ndarray):
            self.v = theano.shared(value=v, borrow=True)
        elif isinstance(v, theano.tensor.TensorVariable):
            self.v = v
        else:
            self.error_("unknown type '%s' for NNArray constructor" % type(v))

    def value(self):
        return self.v

    def shape(self):
        return self.shp

    def update(self, new_value):
        self.assert_(isinstance(new_value, numpy.ndarray),
                     "inconsistent new value '%s' for NNArray" % type(new_value))

        shape1 = self.shape()
        shape2 = new_value.shape
        self.assert_(
            len(shape1) == len(shape2) and
            all([shape1[i] == shape2[i]
                 for i in range(len(shape1))
                 ]),
            "inconsistent shape of new value, %s but %s expected" % (shape1, shape2))

        v = self.v  # ugly warning erasing
        if hasattr(v, "set_value"):
            v.set_value(new_value)
        else:
            self.error_("invalid update for non-shared target value")

    def raw(self):
        v = self.v  # ugly warning erasing
        if hasattr(v, "get_value"):
            return v.get_value()
        else:
            self.error_("only durable NNArray object has raw value")

    def has_raw(self):
        return hasattr(self.v, "get_value")


class NNArrayInt64(NNArray):
    def __init__(self, shape, v=None):
        NNArray.__init__(self, shape, v)


class NNArrayFloat32(NNArray):
    def __init__(self, shape, v=None):
        NNArray.__init__(self, shape, v)


class Layer(NNValue):
    pass


class Loss(NNValue):
    pass


class LogisticLayer(Layer):
    def __init__(self, name, inputs, params):
        Layer.__init__(self)

        self.params = params
        self.name = name

        if len(inputs) < 2:
            self.error_("logistic layer '%s' need more inputs: [x, W, b]" % self.name)
        if not all([isinstance(x, NNValue) for x in inputs]):
            self.error_("inputs for layer '%s' should be of 'NNValue' type" % self.name)

        self.x = inputs[0]
        self.W = inputs[1]

        if len(self.x.size()) != 2:
            self.error_("input data for logistic layer '%s' should be of dimension 2" % self.name)
        if len(self.W.size()) != 1:
            self.error_("weight vector for logistic layer '%s' should be of dimension 1" % self.name)

        if len(inputs) > 2:
            self.b = inputs[2]
        else:
            self.b = NNScalarFloat32(0.)

        self.y = NNArrayFloat32(
            v=theano.tensor.nnet.sigmoid(
                theano.tensor.dot(self.x.value(), self.W.value()) + self.b.value()
            ),
            shape=[self.x.size()[0]]
        )

    def value(self):
        return self.y.value()

    def node(self, ref=None):
        if ref is None:
            return self.y
        else:
            self.error_("logistic layer '%s' has no attribute '%s'" % (self.name, ref))


class SoftmaxLayer(Layer):
    def __init__(self, name, inputs, params):
        Layer.__init__(self)

        self.name = name
        self.params = params
        self.x = inputs[0].value()
        self.W = inputs[1].value()
        self.W_shape = inputs[1].shape()
        self.b = 0.
        if len(inputs) > 2:
            self.b = inputs[2]
        self.y = theano.tensor.nnet.softmax(
            theano.tensor.dot(self.x, self.W) + self.b
        )

    def value(self):
        return self.y

    def node(self, ref=None):
        return self

    def shape(self):
        return [None, self.W_shape[1]]


# 句子卷积层
# 输入：[ data: batch_size * sentence_length * wordvec_length,
#        kernel: kernel_depth * 1 * kernel_width * wordvec_length,
#        mask: batch_size * sentence_length ]
# 输出：batch_size * (sentence_length-kernel_width+1) * kernel_depth
class SentenceConvLayer(Layer):
    def __init__(self, name, inputs, params):
        Layer.__init__(self)

        self.name = name
        self.params = params

        if len(inputs) != 3:
            self.error_("sentence convolution layer '%s' need 3 inputs: [data, kernel, mask]" % self.name)
        if not all([isinstance(x, NNArray) for x in inputs]):
            self.error_("inputs for layer '%s' should be of 'NNValue' type" % self.name)

        self.data, self.mask, self.kernel = inputs

        if len(self.data.shape()) != 3:
            self.error_("input data for convolution layer '%s' should be of dimension 3" % self.name)
        if len(self.kernel.shape()) != 3:
            self.error_("kernel for convolution layer '%s' should be of dimension 3" % self.name)
        if not all([isinstance(x, int) for x in self.kernel.shape()]):
            self.error_("kernel shape unknown for convolution layer '%s'" % self.name)
        if len(self.mask.shape()) != 2:
            self.error_("data mask for convolution layer '%s' should be of dimension 2" % self.name)

        filter_shape = self.kernel.shape()[0:1] + [1] + self.kernel.shape()[1:]

        raw_conv = theano.tensor.nnet.conv.conv2d(
            input=self.data.value().dimshuffle(0, 'x', 1, 2),
            filters=self.kernel.value().dimshuffle(0, 'x', 1, 2),
            filter_shape=filter_shape
        )

        kernel_width = self.kernel.shape()[1]
        conv_mask = self.mask.value()
        if kernel_width > 1:
            conv_mask = conv_mask[:, :1 - kernel_width]
        else:
            self.error_("kernel width of convolution layer '%s' should be at least 1" % self.name)

        reduced_conv = theano.tensor.nnet.sigmoid(
            theano.tensor.sum(raw_conv, axis=3)
        ).dimshuffle(0, 2, 1) * conv_mask.dimshuffle(0, 1, 'x')

        data_size = self.data.shape()
        trim_sentence_length = None if not data_size[1] else data_size[1] - kernel_width + 1
        self.out = NNArrayFloat32(v=reduced_conv,
                                  shape=[data_size[0], trim_sentence_length, data_size[2]])

    def value(self):
        return self.out.value()

    def node(self, ref=None):
        if ref is None:
            return self.out
        else:
            self.error_("convolution layer '%s' has no attribute '%s'" % (self.name, ref))


# MaxPooling层
# 输入：[ data: batch_size * X * Y ]
# 输出：batch_size * Y
class MaxPoolingLayer(Layer):
    def __init__(self, name, inputs, params):
        Layer.__init__(self)

        self.name = name
        self.params = params

        if len(inputs) != 1:
            self.error_("max-pooling layer '%s' requires exactly 1 input" % self.name)
        if not all([isinstance(x, NNArray) for x in inputs]):
            self.error_("inputs for layer '%s' should be of 'NNArray' type" % self.name)

        self.data = inputs[0]
        if len(self.data.shape()) != 3:
            self.error_("data for max-pooling layer '%s' should be of dimension 3" % self.name)

        pooling = theano.tensor.max(self.data.value(), axis=1)
        self.out = NNArrayFloat32(v=pooling,
                                  shape=[self.data.shape()[0], self.data.shape()[2]])

    def node(self, ref=None):
        if ref is None:
            return self.out
        else:
            self.error_("pooling layer '%s' has no attribute '%s'" % (self.name, ref))

    def value(self):
        return self.out.value()


class BinaryCrossEntropyLoss(Loss):
    def __init__(self, name, inputs, params):
        Loss.__init__(self)

        self.name = name
        self.params = params

        if len(inputs) != 2:
            self.error_("BinaryCrossEntropy loss need inputs: [predict, label]")
        if not all([isinstance(x, NNValue) for x in inputs]):
            self.error_("inputs for loss '%s' should be of 'NNValue' type" % self.name)

        self.x, self.y = inputs[0], inputs[1]

        if len(self.x.size()) != 1 or len(self.y.size()) != 1:
            self.error_("input data for BinaryCrossEntropy loss should be of dimension 1")

        x, y = self.x.value(), self.y.value()
        loss = -theano.tensor.mean(
            x * theano.tensor.log(y) + (1 - x) * theano.tensor.log(1 - y)
        )
        predict = theano.tensor.switch(theano.tensor.gt(x, 0.5), 1, 0)
        self.loss = NNScalarFloat32(loss)
        self.predict = NNArrayInt64(shape=self.x.size()[:1], v=predict)

    def node(self, ref=None):
        if ref is None:
            return self
        elif ref == "predict":
            return self.predict
        else:
            self.error_("BinaryCrossEntropy loss has no attribute '%s'" % ref)

    def value(self):
        return self.loss.value()


class CrossEntropyLoss(Loss):
    def __init__(self, name, inputs, params):
        Loss.__init__(self)

        self.name = name
        self.params = params

        if len(inputs) != 2:
            self.error_("CrossEntropy loss's inputs are: [predict, label]")

        self.assert_(all([isinstance(_, NNValue) for _ in inputs]),
                     "inputs for loss '%s' should be of NNValue" % self.name)

        self.x = inputs[0]
        self.y = inputs[1]

        if len(self.x.shape()) != 2 or len(self.y.shape()) != 2:
            self.error_("input shape for BinaryCrossEntropy loss should [sampleNum, classNum]")

        x, y = self.x.value(), self.y.value()
        loss = -theano.tensor.mean(
            theano.tensor.sum(
                y * theano.tensor.log(x), axis=1
            )
        )
        predict = theano.tensor.argmax(
            x, axis=1
        )
        self.loss = NNScalarFloat32(loss)
        self.predict = NNArrayInt64(shape=self.x.shape()[:1], v=predict)

    def node(self, ref=None):
        if ref is None:
            return self
        elif ref == "predict":
            return self.predict
        else:
            self.error_("CrossEntropy loss has no attribute '%s'" % ref)

    def value(self):
        return self.loss.value()


class Update(object):
    def __init__(self):
        self.updates = None

    @staticmethod
    def error_(s):
        NNDebug.error_(("[Update] ", s))

    @staticmethod
    def assert_(cond, s):
        NNDebug.assert_(cond, s)

    def get(self):
        """
        :return: NNValue update function
        """
        return self.updates


class SGDUpdate(Update):
    def __init__(self, loss, ws, params):
        Update.__init__(self)

        if not isinstance(loss, NNValue):
            self.error_("invalid loss type for SGDUpdata: %s" % loss)
        
        if params is None:
            params = {}

        self.loss = loss
        self.grads = dict([
            (name,
             theano.tensor.grad(self.loss.value(), ws[name].value())
             )
            for name in ws
        ])
        self.updates = map(
            lambda (w, g): (w, w - params.get("learning_rate", 0.1) * g),
            [(ws[n].value(), self.grads[n]) for n in ws]
        )

    def get(self):
        return self.updates


class NNBase:

    def __init__(self):

        self.__core = {

            # training function
            "train_func": None,

            # testing function
            "test_func": None,

            # input NNValue dict
            "inputs": {},

            # weight NNValue dict
            "weights": {},

            # weight to learn NNValue dict
            "learn": {},

            # update function
            "updates": None,

            # output NNValue dict of training process
            "train_info": None,

            # output NNValue dict of testing process
            "test_info": None,

            # data NNValue dict
            "data": {},

            # dict for all NNValue
            "values": {},

            # loss NNValue node
            "loss": None
        }

    @staticmethod
    def __error(text):
        raise Exception(text)

    @staticmethod
    def __assert(cond, msg):
        NNDebug.assert_(cond, msg)

    def __make_train_test(self):

        i = theano.tensor.lscalar()
        j = theano.tensor.lscalar()

        # no need to check data existence
        givens = dict([
            (v.value(), self.__core["data"][name].value()[i:j])
            for name, v in self.__core["inputs"].items()
        ])

        try:
            self.__core["train_func"] = theano.function(
                inputs=[i, j],
                givens=givens,
                updates=self.__core["updates"].get(),
                outputs=[v.value() for (n, v) in self.__core["train_info"]]
            )

            self.__core["test_func"] = theano.function(
                inputs=[i, j],
                givens=givens,
                outputs=[v.value() for (n, v) in self.__core["test_info"]]
            )
        except theano.compile.UnusedInputError:
            self.__error("there are unused input in your network")

    def __check_name(self, name, cat="values"):
        if name not in self.__core.get(cat, "values"):
            self.__error("'%s' not found in network %s" % (name, cat))

    def __check_data(self, name, data):
        self.__check_name(name, "inputs")
        self.__assert(isinstance(data, numpy.ndarray), "data should be numpy.ndarray" % name)
        value = self.__core["inputs"][name]
        self.__assert(isinstance(value, NNArray), "'%s' should be input NNArray" % name)
        data_shape = data.shape
        node_shape = value.shape()
        if len(data_shape) != len(node_shape) or \
            not all([(y is None or x == y)
                    for x, y in zip(data_shape, node_shape)]):
            self.__error("inconsistent shape %s for input '%s', %s required" % (data_shape, name, node_shape))

    def __check_func(self):
        for name in self.__core["inputs"]:
            self.__check_name(name, "inputs")
            self.__check_name(name, "data")
        if self.__core["train_func"] is None:
            self.__make_train_test()

    @staticmethod
    def __default_collector(v1, v2):
        if isinstance(v1, numpy.ndarray):
            if v1.shape == ():
                return numpy.array(v1 + v2)
            else:
                return numpy.concatenate([v1, v2])
        else:
            return None

    def train(self, beg, end, iters=None, batch=None,
              iter_reporter=None, batch_collector=None):
        """
        training method

        :param beg: int, start offset of all provided input data

        :param end: int, end offset of all provided input data

        :param iters: int, training iterations, defaults to 1

        :param batch: int, training batch size, defaults to (end-beg)

        :param iter_reporter: callback function, called on each iteration finish,
                            with the result of this iteration passed as parameter

        :param batch_collector: callback function, determine how result from
                            different batches get merged, by default, scalar
                            values are added up and arrays are concatenated

        :return: results specified in "train_info" configuration section, the results
                of the last iteration are returned if iters>1
        """
        self.__check_func()
        train_func = self.get_core()["train_func"]

        if iter_reporter is not None:
            self.__assert(hasattr(iter_reporter, "__call__"),
                          "iter_reporter for train() should be callable")
        if batch_collector is not None:
            self.__assert(hasattr(batch_collector, "__call__"),
                          "batch_collector for train() should be callable")
        else:
            batch_collector = self.__default_collector

        if iters is None:
            iters = 1
        if batch is None:
            batch = end - beg
        batch_num = (end-beg) / batch + 1

        result = None

        for i in range(iters):
            result = None
            for j in range(batch_num):
                l = j * batch
                r = (j+1) * batch
                if l >= end:
                    break
                if r > end:
                    r = end
                new_result = train_func(l, r)

                if result is None:
                    result = new_result
                else:
                    result = [batch_collector(a, b)
                              for a, b in zip(result, new_result)]

            if iter_reporter is not None:
                try:
                    iter_reporter(i, result)
                except Exception as e:
                    self.__error("exception in iter_reporter '%s'" % e)

        return result

    def test(self, beg, end, batch=None, batch_collector=None):
        """
        testing method

        :param beg: int, start offset of all provided input data

        :param end: int, end offset of all provided input data

        :param batch: int, test batch size, defaults to (end-beg)

        :param batch_collector: callback function, determine how result from
                            different batches get merged, by default, scalar
                            values are added up and arrays are concatenated

        :return: results specified in "test_info" configuration section, the results
                of the last iteration are returned if iters>1
        """
        self.__check_func()
        test_func = self.get_core()["test_func"]

        if batch_collector is not None:
            self.__assert(hasattr(batch_collector, "__call__"),
                          "batch_collector for test() should be callable")
        else:
            batch_collector = self.__default_collector

        if batch is None:
            batch = end - beg
        batch_num = (end-beg) / batch + 1

        result = None
        for j in range(batch_num):
            l = j * batch
            r = (j+1) * batch
            if l >= end:
                break
            if r > end:
                r = end
            new_result = test_func(l, r)

            if result is None:
                result = new_result
            else:
                result = [batch_collector(a, b)
                          for a, b in zip(result, new_result)]
        return result

    def get_core(self):
        return self.__core

    def get_data(self, name):
        """
        retrieve the input data of the network, raise exception
        if name not found in inputs

        :param name: str
        :return: numpy.ndarray
        """
        self.__check_name(name, cat="data")
        return self.get_core()["data"].raw()

    def set_data(self, name, data):
        """
        update the input data of the network, raise exception
        if name not found in inputs or inconsistent new data

        :param name: str
        :param data: numpy.ndarray, new data to be set
        """
        self.__check_name(name, cat="data")
        self.__check_data(name, data)
        self.get_core()["data"].update(data)

    def get_value(self, name):
        """
        retrieve the raw value for network node/param, raise
        exception if name not found or no raw value

        :param name: str
        :return: raw value,
        """
        self.__check_name(name)
        value = self.get_core()["values"][name]
        return value.raw()

    def set_value(self, name, new_value):
        """
        update the value for network, raise exception if name
        not found or inconsistent new value

        :param name: str
        :param new_value: new value
        """
        self.__check_name(name)
        value = self.__core["values"][name]
        value.update(new_value)

    def load(self, path, weights=None):
        """
        load network weights from model file

        :param path: str, file path
        :param weights:
            if None, load all weights from file;
            if list, the weight names which will load
            if dict, the (name:bias) items where bias
                    is the name used in the file
        """
        try:
            input_file = open(path, "rb")
            try:
                data_dict = numpy.load(input_file).item()
                if not isinstance(data_dict, dict):
                    raise IOError()
            except IOError:
                self.__error("bad format file '%s'" % path)
            finally:
                input_file.close()
        except IOError:
            self.__error("'%s' not found" % path)

        if isinstance(weights, dict):
            pass
        if isinstance(weights, list):
            weights = dict([(n, n) for n in weights])
        else:
            weights = dict([(n, n) for n in self.get_core()["weights"]])

        for name, bias in weights.items():
            if bias not in data_dict:
                self.__error("missing '%s' in input data" % weights[bias])
            if name not in self.get_core()["weights"]:
                continue
            self.set_value(name, data_dict[bias])

    def save(self, path, weights=None):
        """
        save network weights to model file

        :param path: str, file path
        :param weights:
            if None, save all weights to file;
            if list, the weight names which will save
            if dict, the (name:bias) items where bias
                    is the name used in the file
        """
        if isinstance(weights, dict):
            pass
        if isinstance(weights, list):
            weights = dict([(n, n) for n in weights])
        else:
            weights = dict([(n, n) for n in self.get_core()["weights"]])

        data_dict = {}
        for name, bias in weights.items():
            v = self.get_value(name)
            data_dict[bias] = v

        try:
            output_file = open(path, "wb")
            numpy.save(output_file, data_dict)
            output_file.close()
        except IOError:
            self.__error("save to '%s' failed" % path)
