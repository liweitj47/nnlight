from core.backend import BackendBuilder
from value.values import NNValue
from layer.basic.input import InputLayer
from value.values import NNScalarInt64, NNArrayInt64, NNScalarFloat32, NNArrayFloat32
from utility.debug import NNDebug
from codegen import CodeGenerator
from computation_on_java_network import ComputationOnJavaNetwork


class ComputationOnJavaBackendBuilder(BackendBuilder):

    def __init__(self, core):
        BackendBuilder.__init__(self, core)
        self.binder = ComputationOnJavaBinder(self)
        self.codegen = CodeGenerator()

        # mapping from value to its index for layer
        self.value_index_mapping = {}

        # mapping to "base type"
        self.datatype_mapping = {
            NNScalarFloat32: "double",
            NNScalarInt64: "long",
            NNArrayFloat32: "double",
            NNArrayInt64: "long"
        }

    @staticmethod
    def error(msg):
        NNDebug.error("[java builder] " + msg)

    def build(self):
        self.header()
        self.computation_class()
        code = self.codegen.code()
        return ComputationOnJavaNetwork(code)

    def header(self):
        code = self.codegen
        code.importt("java.lang.Math")
        code.append()

    def output_struct(self):
        code = self.codegen
        code.append()
        code.begin_class("OutputRecord")
        fields = []
        for value in self.core.output_target:
            name = code.escape(self.core.output_name_mapping[value])
            _, field_type = self.binder.get(value)
            fields.append((field_type, name))
            code.field(field_type, name, scope="public")
        code.append()
        code.begin_function("", "OutputRecord", fields)
        for _, name in fields:
            code.assignment("this." + name, name)
        code.end()
        code.end()
        code.append()

    def computation_function(self):
        code = self.codegen
        inputs = []
        for name, inp in self.core.inputs.items():
            name = code.escape(name)
            inputs.append((self.get_type(inp.get_value()), name))
        code.begin_function("OutputRecord", "compute", inputs)
        for layer in self.core.tps_list:
            funcname = self.layer_computation_funcname(layer)
            params = []
            if isinstance(layer, InputLayer):
                params.append(code.escape(layer.name))
            code.call(funcname, params)
        outputs = []
        for value in self.core.output_target:
            outputs.append("this." + self.binder.get_name(value))
        code.new("record", "OutputRecord", params=outputs)
        code.returnn("record")
        code.end()
        code.append()

    def members(self):
        code = self.codegen
        code.append()
        for value in self.binder:
            name, typ = self.binder.get(value)
            basetype = self.binder.get_base_type(value)
            if len(value.get_shape()) > 0 and not isinstance(value.get_father(), InputLayer):
                code.new_array(name, basetype, dims=value.get_shape(), scope="private")
            else:
                code.field(typ, name, scope="private")
            code.append()

    def computation_class(self):
        code = self.codegen
        classname = "Network"
        code.begin_class(classname)
        self.output_struct()
        self.computation_function()
        self.load_function()
        self.layer_computation_functions()
        self.members()
        code.end()

    def load_function(self):
        code = self.codegen
        code.begin_function("void", "load", [])
        code.end()
        code.append()

    def layer_computation_functions(self):
        code = self.codegen
        for layer in self.core.tps_list:
            if not hasattr(layer, "get_computation_on_java_code"):
                self.error("missing 'get_computation_on_java_code()' method for %s instance '%s' "
                           "to support java computation" % (layer.__class__.__name__, layer.name))
            params = []
            if isinstance(layer, InputLayer):
                params.append((self.get_type(layer.get_value()), code.escape(layer.name)))
            code.begin_function("void", self.layer_computation_funcname(layer),
                                params=params, scope="private")
            layer.get_computation_on_java_code(self.codegen, self.binder)
            code.end()
            code.append()

    def layer_computation_funcname(self, layer):
        return "compute_" + self.codegen.escape(layer.name)

    def get_type(self, value):
        base = self.datatype_mapping[value.__class__]
        dims = str.join("", ["[]" for _ in value.get_shape()])
        return base + dims

    def bind_value(self, value):
        layer = value.get_father()
        mapping = self.value_index_mapping.setdefault(layer, {})
        index = len(mapping)
        mapping[value] = index
        name = self.codegen.escape(layer.name) + "$" + str(index)
        typ = self.get_type(value)
        return name, typ

    def get_constructor_map(self):
        """
        :return: versions to support computation on java
        """
        from layers.input import InputLayer
        from layers.logistic import LogisticLayer
        from layers.lstm import LstmLayer
        from layers.pooling import MaxPoolingWithTimeLayer
        from layers.sentence_conv import SentenceConvolutionLayer
        from layers.simple import SimpleLayer
        from layers.softmax import SoftmaxLayer, SequencialSoftmaxLayer
        from layers.tensor import LowRankTensorLayer
        from layers.weight import WeightLayer
        from losses.cross_entropy import CrossEntropyLoss, BinaryCrossEntropyLoss
        from losses.cross_entropy import SequentialCrossEntropyLoss, SequentialBinaryCrossEntropyLoss
        from losses.max_margin import MaxMarginLoss
        return {
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
        }


class ComputationOnJavaBinder:

    def __init__(self, builder):
        self.builder = builder
        self.mapping = {}

    def get(self, value):
        """
        :param value: NNValue object
        :return: tuple (name, type)
        """
        if not isinstance(value, NNValue):
            NNDebug.error("[java builder] ComputationOnJavaBinder.get() acquire NNValue instance")

        if value in self.mapping:
            return self.mapping[value]
        else:
            if not value.get_father():
                NNDebug.error("[java builder] missing father field for NNValue "
                              "passed to ComputationOnJavaBinder.get()")
            else:
                info = self.builder.bind_value(value)
                self.mapping[value] = info
                return info

    def get_name(self, value):
        return self.get(value)[0]

    def get_type(self, value):
        return self.get(value)[1]

    def get_base_type(self, value):
        typ = self.get_type(value)
        idx = typ.find("[")
        return typ[:idx] if idx >= 0 else typ

    def __iter__(self):
        return self.mapping.__iter__()
