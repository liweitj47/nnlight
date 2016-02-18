import importlib
from layer.basic.input import InputLayer
from layer.basic.logistic import LogisticLayer
from layer.basic.lstm import LstmLayer
from layer.basic.pooling import MaxPoolingWithTimeLayer
from layer.basic.sentence_conv import SentenceConvolutionLayer
from layer.basic.simple import SimpleLayer
from layer.basic.softmax import SoftmaxLayer, SequencialSoftmaxLayer
from layer.basic.tensor import LowRankTensorLayer
from layer.basic.weight import WeightLayer
from loss.basic.cross_entropy import CrossEntropyLoss, BinaryCrossEntropyLoss
from loss.basic.cross_entropy import SequentialCrossEntropyLoss, SequentialBinaryCrossEntropyLoss
from loss.basic.max_margin import MaxMarginLoss
from updater.updaters import SGDUpdater
from utility.debug import NNDebug
from value.values import NNScalarInt64, NNArrayFloat32, NNScalarFloat32

cm = {
        # value
        NNScalarInt64: ["int64", "int"],
        NNScalarFloat32: ["float32", "float"],
        NNArrayFloat32: ["[float32]", "[float]"],

        # general
        InputLayer: ["input"],
        WeightLayer: ["weight"],

        # layer
        SimpleLayer: ["simple"],
        SentenceConvolutionLayer: ["sentence_convolution"],
        MaxPoolingWithTimeLayer: ["max_pooling"],
        SoftmaxLayer: ["softmax"],
        LogisticLayer: ["logistic"],
        LowRankTensorLayer: ["low_rank_tensor"],
        LstmLayer: ["lstm"],
        SequencialSoftmaxLayer: ["sequencial_softmax"],

        # loss
        MaxMarginLoss: ["max_margin_loss"],
        BinaryCrossEntropyLoss: ["binary_cross_entropy"],
        CrossEntropyLoss: ["cross_entropy"],
        SequentialBinaryCrossEntropyLoss: ["sequencial_binary_cross_entropy"],
        SequentialCrossEntropyLoss: ["sequencial_cross_entropy"],

        # parameter updater
        SGDUpdater: ["sgd"]
    }


default_constructor_map = {}
for constr, datatypes in cm.items():
    for datatype in datatypes:
        default_constructor_map[datatype] = constr
constructor_map = dict(default_constructor_map.items())


class Constructor:

    def __init__(self):
        pass

    @staticmethod
    def get_default_constructor_map():
        return default_constructor_map

    @staticmethod
    def get_default_constructor(name):
        return default_constructor_map.get(name)

    @staticmethod
    def get_default_array_constructor(name):
        if name.startswith("["):
            return default_constructor_map.get(name)
        else:
            return default_constructor_map.get("[" + name + "]", None)

    @staticmethod
    def get_constructor(name):
        return constructor_map.get(name, None)

    @staticmethod
    def create_value(father, shape, dtype="float32"):
        if isinstance(shape, int):
            shape = [-1 for _ in range(shape)]
        NNDebug.check(isinstance(shape, list) or isinstance(shape, tuple),
                      "[Constructor] to create NNValue instance, shape for"
                      " create_value() need to be list or tuple")
        if len(shape) == 0:
            constructor = Constructor.get_default_constructor(dtype)
            if constructor is None:
                NNDebug.error("[Constructor] invalid dtype for create_value(): '%s'" % dtype)
            return constructor(father)
        else:
            constructor = Constructor.get_default_array_constructor(dtype)
            if constructor is None:
                NNDebug.error("[Constructor] invalid dtype for create_value(): '%s'" % dtype)
            return constructor(shape, father)

    @staticmethod
    def register_type(name, classpath):
        if name in constructor_map:
            NNDebug.error("[Constructor] typename '%s' already defined" % name)
        pos = classpath.rfind(".")
        if pos <= 0:
            NNDebug.error("[Constructor] invalid classpath '%s'" % classpath)
        modulepath, classname = classpath[:pos], classpath[pos+1:]
        try:
            module = importlib.import_module(modulepath)
            classobj = getattr(module, classname, None)
            if not classobj:
                NNDebug.error("[Constructor] class '%s' not found in module %s" % (classname, modulepath))
            constructor_map[name] = classobj
        except ImportError as e:
            NNDebug.error("[Constructor] import error: %s" % e)