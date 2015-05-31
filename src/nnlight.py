# -*- coding: utf-8 -*-
"""
A portable deep learning framework based on Theano
"""
import random
import theano
import numpy
from theano import tensor as T


class NNLogger:
    def __init__(self, level):
        self.level = level

    def log(self, msg, cls=None):
        if self.level < 4:
            return
        if not cls:
            msg = "[", cls, "] " + msg
        raise Exception(msg)

logger = NNLogger(4)


global_config = {}


def get_config():
    cm = {
        NNScalarInt64: ["int64", "int"],
        NNScalarFloat32: ["float32", "float"],
        NNArrayFloat32: ["[float32]", "[float]"],
        NNArrayInt64: ["[int64]", "[int]"],

        SentenceConvLayer: ["sentence_convolution"],
        MaxPoolingLayer: ["max_pooling"],
        SoftmaxLayer: ["softmax"],
        LogisticLayer: ["logistic"],

        BinaryCrossEntropyLoss: ["binary_cross_entropy"]
    }

    constructor_map = {}
    for constructor, datatypes in cm.items():
        for datatype in datatypes:
            constructor_map[datatype] = constructor
    global_config["constructor_map"] = constructor_map


def create(src, data_dict=None):
    if len(global_config) == 0:
        get_config()
    if data_dict is None:
        data_dict = {}
    if isinstance(src, str):
        return create_from_file(src, data_dict)
    else:
        return create_from_dict(src, data_dict)


def create_from_file(config_path, data_dict=None):
    try:
        config_file = open(config_path)
    except IOError:
        logger.log("unable to open configration file '%s" % config_path)
        return None
    config_file.read()
    d = {}
    return create_from_dict(d, data_dict)


def create_from_dict(d, data_dict=None):

    def raise_(s):
        logger.log(s, "[NN Constructor")

    def eval_(v):
        if isinstance(v, str):
            refs = v.strip().split(".")
            prefix = []

            node = nn.core["values"].get(refs[0], None)
            if not node:
                raise_("'%s' not found" % refs[0])
            prefix.append(refs[0])
            node = node.node()

            for ref in refs[1:]:
                node = node.node(ref=ref)
                if not node:
                    raise_("'%s' not found in '%s'<%s>" % (ref, str.join(".", prefix), node))
                prefix.append(ref)

            return node

        elif isinstance(v, int):
            return NNScalarInt64(v)
        elif isinstance(v, float):
            return NNScalarFloat32(v)
        elif isinstance(v, NNValue):
            return v
        else:
            raise_("unsupported value '%s'" % v)

    def raw_(v):
        if isinstance(v, NNValue):
            return v.value()
        elif isinstance(v, list):
            return map(raw_, v)
        else:
            return v

    def make_input_(dims, datatype):
        if isinstance(dims, int):
            dims = [None for _ in range(dims)]
        if isinstance(dims, list):
            if len(dims) == 0:
                constructor = config["constructor_map"].get(datatype, None)
                if constructor:
                    return constructor()
            else:
                datatype = "[" + datatype + "]"
                constructor = config["constructor_map"].get(datatype, None)
                if constructor:
                    return constructor(dims=dims)
        raise_("unsupported datatype '%s'" % datatype)

    def make_shared_(dim_size, init_method, datatype):
        basetype = datatype
        datatype = "[" + basetype + "]"
        dim_size = map(eval_, dim_size)

        if not (
            isinstance(dim_size, list) and
            len(dim_size) > 0 and
            all([isinstance(_, NNScalarInt) for _ in dim_size]) and
            all([_.is_durable() for _ in dim_size])
        ):
            raise_("size must be an integer array")
        if basetype not in ["float32", "int64"]:
            raise_("data type '%s' not supported for weights" % datatype)
        if datatype not in config["constructor_map"]:
            raise_("data type '%s' not supported for weights" % datatype)

        dim_size = map(raw_, dim_size)
        v = numpy.zeros(dim_size, dtype=basetype)
        if init_method == "random":
            high = 4.0 * numpy.sqrt(6.0 / sum(dim_size))
            low = -high

            def __randomize__(w, dims):
                if len(dims) == 1:
                    for i in range(dims[0]):
                        w[i] = random.uniform(low, high)
                else:
                    for i in range(dims[0]):
                        __randomize__(w[i], dims[1:])

            __randomize__(v, dim_size)

        constructor = config["constructor_map"][datatype]
        return constructor(dims=dim_size, v=NNData(v))

    def make_layer_(layer):
        lname = layer.get("name")
        ltype = layer.get("type", None)
        if not ltype:
            raise_("layer must be explicitly assigned type")

        constructor = config["constructor_map"].get(ltype)
        if not constructor:
            raise_("unsupported layer type '%s'" % ltype)

        inputs = map(eval_, layer.get("input", []))
        params = dict([(k, eval_(v)) for k, v in layer.get("param", {}).items()])
        return constructor(lname, inputs, params)

    def make_loss_(loss):
        lname = loss.get("name")
        ltype = loss.get("type", None)
        if not ltype:
            raise_("loss must be explicitly assigned type")

        constructor = config["constructor_map"].get(ltype, None)
        if not constructor:
            raise_("loss type '%s' not supported" % ltype)

        inputs = map(eval_, loss.get("input", []))
        params = dict([(k, eval_(v)) for k, v in loss.get("param", {}).items()])
        return constructor(lname, inputs, params)

    def wrap_data_(data):
        if data is None:
            return None
        elif isinstance(data, numpy.ndarray):
            return NNData(data)
        else:
            raise_("type '%s' not supported for input data" % type(data))

    def make_update_(loss, ws, params):
        mappings = {
            "sgd": SGDUpdate
        }
        gradtype = params.get("method", "sgd")
        return mappings[gradtype](loss, ws, params)

    def check_name_(s):
        if not isinstance(s, str) or s == "":
            return False
        elif s in nn.core["values"]:
            raise_("duplicate definition '%s'" % s)
        else:
            return True

    if data_dict is None:
        data_dict = {}
    nn = NNBase()
    config = global_config

    if not isinstance(data_dict, dict):
        raise_("data_dict should be a dictionary")

    # parameters
    for item in d.get("param", []):
        name = item.get("name", "")
        if not check_name_(name):
            continue
        value = item.get("value", None)
        if not value:
            continue
        nn.core["values"][name] = eval_(value)

    # inputs
    for item in d.get("input", []):
        name = item.get("name", "")
        if not check_name_(name):
            continue
        dim = item.get("dim")
        dtype = item.get("type", "float32")
        value = make_input_(dim, dtype)
        nn.core["values"][name] = value
        nn.core["inputs"][name] = value
        nn.core["data"][name] = wrap_data_(data_dict.get(name, None))

    # weights
    for item in d.get("weight", []):
        name = item.get("name", "")
        if not check_name_(name):
            continue
        size = item.get("size", [])
        init = item.get("init", "random")
        dtype = item.get("type", "float32")
        value = make_shared_(size, init, dtype)
        nn.core["values"][name] = value
        nn.core["weights"][name] = value
        if item.get("update", True):
            nn.core["learn"][name] = value

    # layers
    for item in d.get("layer", []):
        name = item.get("name", "")
        if not check_name_(name):
            continue
        nn.core["values"][name] = make_layer_(item)

    # losses
    for item in d.get("loss", []):
        name = item.get("name", "")
        if not check_name_(name):
            continue
        value = make_loss_(item)
        nn.core["values"][name] = value

    # training
    item = d.get("training", None)
    if not item:
        raise_("missing training section")
    if "loss" not in item:
        raise_("missing loss function")
    nn.core["loss"] = eval_(item["loss"])
    nn.core["updates"] = make_update_(
        nn.core["loss"],
        nn.core["learn"],
        {
            "learning_rate": item.get("learning_rate", 0.1),
            "method": item.get("method", "sgd")
        }
    )
    nn.core["test_info"] = [(n, eval_(n))
                            for n in filter(lambda _: isinstance(_, str), item.get("test_info", []))]
    nn.core["train_info"] = [(n, eval_(n))
                             for n in filter(lambda _: isinstance(_, str), item.get("train_info", []))]
    return nn


class NNData:
    def __init__(self, data):
        self.data = theano.shared(value=data, borrow=True)

    def get(self):
        return self.data.get_value()

    def get_wrap(self):
        return self.data


class NNValue(object):
    @staticmethod
    def raise_(s):
        logger.log(s, "[NNValue]")

    def __init__(self):
        self.v = None
        self.durable = False
        self.dims = None

    def node(self, ref=None):
        return self

    def value(self):
        return self.v

    def is_durable(self):
        if not hasattr(self, "durable"):
            setattr(self, "durable", False)
        return self.durable


class NNScalar(NNValue):
    pass


class NNScalarInt(NNScalar):
    pass


class NNScalarInt64(NNScalarInt):
    def __init__(self, v=None):
        NNScalarInt.__init__(self)
        if v is None:
            self.v = T.scalar()
            self.durable = False
        else:
            self.v = v
            self.durable = True


class NNScalarFloat32(NNScalar):
    def __init__(self, v=None):
        NNScalar.__init__(self)
        if v is None:
            self.v = T.scalar()
            self.durable = False
        else:
            self.v = v
            self.durable = True


class NNArray(NNValue):
    def value(self):
        if isinstance(self.v, NNData):
            return self.v.get_wrap()
        else:
            return self.v

    def size(self):
        return self.dims


class NNArrayInt64(NNArray):
    def __init__(self, dims, v=None):
        NNArray.__init__(self)
        self.dims = dims
        if v:
            self.v = v
            self.durable = True
        else:
            if len(dims) > 4:
                NNValue.raise_("NNArray support at most 4Darray temporarily")
            self.v = [T.vector, T.matrix, T.tensor3, T.tensor4][len(dims) - 1]()
            self.durable = False


class NNArrayFloat32(NNArray):
    def __init__(self, dims, v=None):
        NNArray.__init__(self)
        self.dims = dims
        if v:
            self.v = v
            self.durable = True
        else:
            if len(dims) > 4:
                NNValue.raise_("NNArray support at most 4Darray temporarily")
            self.v = [T.vector, T.matrix, T.tensor3, T.tensor4][len(dims) - 1]()
            self.durable = False


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
            NNValue.raise_("logistic layer '%s' need more inputs: [x, W, b]" % self.name)
        if not all([isinstance(x, NNValue) for x in inputs]):
            NNValue.raise_("inputs for layer '%s' should be of 'NNValue' type" % self.name)

        self.x = inputs[0]
        self.W = inputs[1]

        if len(self.x.size()) != 2:
            NNValue.raise_("input data for logistic layer '%s' should be of dimension 2" % self.name)
        if len(self.W.size()) != 1:
            NNValue.raise_("weight vector for logistic layer '%s' should be of dimension 1" % self.name)

        if len(inputs) > 2:
            self.b = inputs[2]
        else:
            self.b = NNScalarFloat32(0.)

        self.y = NNArrayFloat32(v=T.nnet.sigmoid(T.dot(self.x.value(), self.W.value()) + self.b.value()),
                                dims=[self.x.size()[0]])

    def value(self):
        return self.y.value()

    def node(self, ref=None):
        if ref is None:
            return self.y
        else:
            NNValue.raise_("logistic layer '%s' has no attribute '%s'" % (self.name, ref))


class SoftmaxLayer(Layer):
    def __init__(self, name, inputs, params):
        Layer.__init__(self)

        self.name = name
        self.params = params
        self.x = inputs[0]
        self.W = inputs[1]
        self.b = 0.
        if len(inputs) > 2:
            self.b = inputs[2]
        self.y = T.nnet.softmax(T.dot(self.x, self.W) + self.b)

    def get_output(self):
        return self.y


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
            NNValue.raise_("sentence convolution layer '%s' need 3 inputs: [data, kernel, mask]" % self.name)
        if not all([isinstance(x, NNArray) for x in inputs]):
            NNValue.raise_("inputs for layer '%s' should be of 'NNValue' type" % self.name)

        self.data, self.mask, self.kernel = inputs

        if len(self.data.size()) != 3:
            NNValue.raise_("input data for convolution layer '%s' should be of dimension 3" % self.name)
        if len(self.kernel.size()) != 4:
            NNValue.raise_("kernel for convolution layer '%s' should be of dimension 4" % self.name)
        if not all([isinstance(x, int) for x in self.kernel.size()]):
            NNValue.raise_("kernel shape unknown for convolution layer '%s'" % self.name)
        if len(self.mask.size()) != 2:
            NNValue.raise_("data mask for convolution layer '%s' should be of dimension 2" % self.name)

        raw_conv = T.nnet.conv.conv2d(input=self.data.value().dimshuffle(0, 'x', 1, 2),
                                      filters=self.kernel.value(),
                                      filter_shape=self.kernel.size())

        kernel_width = self.kernel.size()[2]
        conv_mask = self.mask.value()
        if kernel_width > 1:
            conv_mask = conv_mask[:, :1 - kernel_width]
        else:
            NNValue.raise_("kernel width of convolution layer '%s' should be at least 1" % self.name)

        reduced_conv = T.nnet.sigmoid(T.sum(raw_conv, axis=3)).dimshuffle(0, 2, 1) * conv_mask.dimshuffle(0, 1, 'x')

        data_size = self.data.size()
        trim_sentence_length = None if not data_size[1] else data_size[1] - kernel_width + 1
        self.out = NNArrayFloat32(v=reduced_conv,
                                  dims=[data_size[0], trim_sentence_length, data_size[2]])

    def value(self):
        return self.out.value()

    def node(self, ref=None):
        if ref is None:
            return self.out
        else:
            NNValue.raise_("convolution layer '%s' has no attribute '%s'" % (self.name, ref))


# MaxPooling层
# 输入：[ data: batch_size * X * Y ]
# 输出：batch_size * Y
class MaxPoolingLayer(Layer):
    def __init__(self, name, inputs, params):
        Layer.__init__(self)

        self.name = name
        self.params = params

        if len(inputs) != 1:
            NNValue.raise_("max-pooling layer '%s' requires exactly 1 input" % self.name)
        if not all([isinstance(x, NNArray) for x in inputs]):
            NNValue.raise_("inputs for layer '%s' should be of 'NNArray' type" % self.name)

        self.data = inputs[0]
        if len(self.data.size()) != 3:
            NNValue.raise_("data for max-pooling layer '%s' should be of dimension 3" % self.name)

        pooling = T.max(self.data.value(), axis=1)
        self.out = NNArrayFloat32(v=pooling,
                                  dims=[self.data.size()[0], self.data.size()[2]])

    def node(self, ref=None):
        if ref is None:
            return self.out
        else:
            NNValue.raise_("pooling layer '%s' has no attribute '%s'" % (self.name, ref))

    def value(self):
        return self.out.value()


class BinaryCrossEntropyLoss(Loss):
    def __init__(self, name, inputs, params):
        Loss.__init__(self)

        self.name = name
        self.params = params

        if len(inputs) != 2:
            NNValue.raise_("BinaryCrossEntropy loss need inputs: [predict, label]")
        if not all([isinstance(x, NNValue) for x in inputs]):
            NNValue.raise_("inputs for loss '%s' should be of 'NNValue' type" % self.name)

        self.x, self.y = inputs[0], inputs[1]

        if len(self.x.size()) != 1 or len(self.y.size()) != 1:
            NNValue.raise_("input datas for BinaryCrossEntropy loss should be of dimension 1")

        x, y = self.x.value(), self.y.value()
        loss = -T.mean(x * T.log(y) + (1 - x) * T.log(1 - y))
        predict = T.switch(T.gt(x, 0.5), 1, 0)
        self.loss = NNScalarFloat32(loss)
        self.predict = NNArrayInt64(dims=self.x.size()[:1], v=predict)

    def node(self, ref=None):
        if ref is None:
            return self
        elif ref == "predict":
            return self.predict
        else:
            NNValue.raise_("BinaryCrossEntropy loss has no attribute '%s'" % ref)

    def value(self):
        return self.loss.value()


class Update(NNValue):
    def node(self, ref=None):
        NNValue.raise_("node() not supported for Update temporarily")

    def value(self):
        NNValue.raise_("value() not supported for Update temporarily")


class SGDUpdate(Update):
    def __init__(self, loss, ws, params):
        Update.__init__(self)

        if not isinstance(loss, NNValue):
            NNValue.raise_("invalid loss type for SGDUpdata: %s" % loss)
        
        if params is None:
            params = {}

        self.loss = loss
        self.grads = dict([
            (name, T.grad(self.loss.value(), ws[name].value())) for name in ws
        ])
        self.updates = map(
            lambda (w, g): (w, w - params.get("learning_rate", 0.1) * g),
            [(ws[n].value(), self.grads[n]) for n in ws]
        )

    def get(self):
        return self.updates


#################################################################################################


#######################################################################################################

class NNBase:
    def __init__(self):

        self.core = {

            # 训练函数的实现
            "train_func": None,

            # 测试函数的实现
            "test_func": None,

            # 输入 :{str->NNValue}
            "inputs": {},

            # 参数 :{str->NNValue}
            "weights": {},

            # 待学习的参数 :{str->NNValue}
            "learn": {},

            "updates": None,

            # 训练函数的输出 :{str->NNValue}
            "train_info": None,

            # 测试函数的输出 :{str->NNValue}
            "test_info": None,

            # 数据 :{str->NNValune}
            "data": {},

            # 中间数据 :{str->NNValue}
            "values": {},

            "loss": None
        }

    def __make_train_test(self):

        i, j = T.lscalar(), T.lscalar()

        # 不必检查data是否存在
        givens = dict([
            (v.value(), self.core["data"][name].get_wrap()[i:j])
            for name, v in self.core["inputs"].items()
        ])

        self.core["train_func"] = theano.function(
            inputs=[i, j],
            givens=givens,
            updates=self.core["updates"].get(),
            outputs=[v.value() for (n, v) in self.core["train_info"]]
        )

        self.core["test_func"] = theano.function(
            inputs=[i, j],
            givens=givens,
            outputs=[v.value() for (n, v) in self.core["test_info"]]
        )

    @staticmethod
    def __raise(text, level=0):
        raise Exception(text)

    def __check(self):
        # 检查数据
        for name in self.core["inputs"]:
            if name not in self.core["values"]:
                self.__raise("[NNBase] input '%s' not found" % name)
            if name not in self.core["data"]:
                self.__raise("[NNBase] missing input data '%s', use set_data() to set it" % name)
        # 构造训练及测试函数
        if self.core["train_func"] is None:
            self.__make_train_test()

    def train(self, beg, end):
        self.__check()
        return self.core["train_func"](beg, end)

    def test(self, beg, end):
        self.__check()
        return self.core["test_func"](beg, end)
