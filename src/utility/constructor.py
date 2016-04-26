import importlib
from layer.basic.input import InputLayer
from layer.basic.aggregate import ConcatLayer
from layer.basic.transform import DimShuffle
from layer.basic.logistic import LogisticLayer
from layer.basic.pooling import MaxPoolingWithTimeLayer
from layer.basic.sentence_conv import SentenceConvolutionLayer
from layer.basic.simple import SimpleLayer
from layer.basic.softmax import SoftmaxLayer
from layer.basic.tensor import LowRankTensorLayer
from layer.basic.weight import WeightLayer
from layer.sequence.lstm import LstmLayer
from layer.sequence.gru import GruLayer
from layer.sequence.prediction import SequentialSoftmaxLayer, SequentialLogisticLayer
from layer.sequence.attention import AttentionGruLayer
from layer.sequence.gen_seq import SequentialGeneratorWithContext
from loss.basic.cross_entropy import CrossEntropyLoss, BinaryCrossEntropyLoss
from loss.basic.cross_entropy import SequentialCrossEntropyLoss, SequentialBinaryCrossEntropyLoss
from loss.basic.max_margin import MaxMarginLoss
from updater.updaters import SGDUpdater
from utility.debug import NNDebug
from value.values import NNScalarInt64, NNArrayFloat32, NNScalarFloat32

default_cm = {
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

        # sequential layer
        LstmLayer: ["lstm"],
        GruLayer: ["gru"],
        AttentionGruLayer: ["attention_gru"],
        SequentialLogisticLayer: ["sequential_logistic"],
        SequentialSoftmaxLayer: ["sequential_softmax"],
        SequentialGeneratorWithContext: ["gen_seq_with_context"],

        # utility layer
        ConcatLayer: ["concatenate"],
        DimShuffle: ["dimshuffle"],

        # loss
        MaxMarginLoss: ["max_margin_loss"],
        BinaryCrossEntropyLoss: ["binary_cross_entropy"],
        CrossEntropyLoss: ["cross_entropy"],
        SequentialBinaryCrossEntropyLoss: ["sequential_binary_cross_entropy"],
        SequentialCrossEntropyLoss: ["sequential_cross_entropy"],

        # parameter updater
        SGDUpdater: ["sgd"]
    }


default_constructor_map = {}
constructor_map = {}


class Constructor:

    def __init__(self):
        pass

    @staticmethod
    def load_default_constructor_map(reversed_cm):
        default_constructor_map.clear()
        for constr, datatypes in reversed_cm.items():
            for datatype in datatypes:
                default_constructor_map[datatype] = constr

    @staticmethod
    def load_constructor_map(reversed_cm):
        # constructor_map.clear()
        for constr, datatypes in reversed_cm.items():
            for datatype in datatypes:
                constructor_map[datatype] = constr

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
        else:
            shape = [_ for _ in shape]  # copy to avoid latent alias problem
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
    def load_module_from_path(modulepath):
        try:
            module = importlib.import_module(modulepath)
            return module
        except ImportError as e:
            NNDebug.error("[Constructor] import error: %s" % e)

    @staticmethod
    def load_constructor_from_path(classpath):
        pos = classpath.rfind(".")
        if pos <= 0:
            NNDebug.error("[Constructor] invalid classpath '%s'" % classpath)
        modulepath, classname = classpath[:pos], classpath[pos+1:]
        module = Constructor.load_module_from_path(modulepath)
        classobj = getattr(module, classname, None)
        if not classobj:
            NNDebug.error("[Constructor] class '%s' not found in module %s" % (classname, modulepath))
        return classobj

    @staticmethod
    def register_type(name, classpath, constructor=None):
        if name in constructor_map:
                NNDebug.error("[Constructor] typename '%s' already defined" % name)
        if constructor:
            constructor_map[name] = constructor
        else:
            classobj = Constructor.load_constructor_from_path(classpath)
            constructor_map[name] = classobj


Constructor.load_default_constructor_map(default_cm)
Constructor.load_constructor_map(default_cm)
