# -*- coding: utf-8 -*-
"""

"""
from core import NNCore
from layer.layers import Layer
from loss.loss import Loss
from network import Network
from theano_impl.theano_builder import TheanoBackendBuilder
from updater.updaters import Updater
from utility.configparser import ConfigParser
from utility.constructor import Constructor
from utility.debug import NNDebug
from value.values import NNValue


class NNBuilder:

    def __init__(self):
        self.valid_input_dtypes = {"float32", "int64"}
        self.valid_weight_dtypes = {"float32"}
        self.core = NNCore()
        self.names = set()
        self.values = {}

    @staticmethod
    def error(s):
        NNDebug.error("[builder] " + str(s))

    @staticmethod
    def check(cond, s):
        NNDebug.check(cond, "[builder] " + str(s))

    def build(self, src, data_dict=None):
        """
        build the network object from configuration source and
        assign data to the network

        :param src:
            if str, the configuration file path
            if dict, configuration python dictionary

        :param data_dict:
            dict, the key is the network input name defined by
                the configuration, and the value is the
                numpy.ndarray fed to the network's input

        :return: the Network object
        """
        if data_dict is None:
            data_dict = {}
        if isinstance(src, str):
            return self.build_from_file(src, data_dict)
        else:
            return self.build_from_dict(src, data_dict)

    def build_from_file(self, config_path, data_dict=None):
        """
        build the network object from configuration file and
        assign data to the network

        :param config_path:
            str, the configuration file path

        :param data_dict:
            dict, the key is the network input name defined by
                the configuration, and the value is the
                numpy.ndarray fed to the network's input

        :return: the NNBase network object
        """
        parser = ConfigParser()
        d = parser.parse(config_path)
        return self.build_from_dict(d, data_dict)

    def build_from_dict(self, d, data_dict=None):
        """
        build the network object from configuration dictionary and
        assign data to the network

        :param d:
            dict, the configuration dictionary

        :param data_dict:
            dict, the key is the network input name defined by
                the configuration, and the value is the
                numpy.ndarray fed to the network's input

        :return: the NNBase network object
        """
        # basic objects
        if data_dict is None:
            data_dict = {}
        self.frontend_build(d, data_dict)  # build core
        network = self.backend_build()  # build obj code from core
        return network

    def frontend_build(self, d, data_dict):

        # parameters
        for item in d.get("param", []):
            name = item.get("name", "")
            self.check_name(name)
            value = item.get("value")
            if value is None:
                self.error("missing value for parameter '%s'" % name)
            value = self.eval(value)
            self.check(isinstance(value, int) or isinstance(value, float),
                       "parameter '%s' should be an integer or float")
            self.values[name] = value

        # inputs
        for item in d.get("input", []):
            name = item.get("name", "")
            self.check_name(name)
            shape = item.get("shape")
            if shape is None:
                self.error("missing shape field for input '%s'" % name)
            elif isinstance(shape, int):
                shape = [-1 for _ in range(shape)]
            self.check(isinstance(shape, list) and len(shape) > 0 and all([isinstance(x, int) for x in shape]),
                       "shape for input '%s' should be non-empty integer array" % name,)
            self.check(name in data_dict,
                       "input '%s' is not provided in the data dictionary" % name)
            dtype = item.get("type", "float32")
            self.make_input(name, shape, dtype, data_dict[name])

        # weights
        for item in d.get("weight", []):
            name = item.get("name", "")
            self.check_name(name)
            shape = self.eval(item.get("shape"))
            if shape is None:
                self.error("missing shape field for weight '%s'" % name)
            self.check(isinstance(shape, list) and all([isinstance(x, int) and x > 0 for x in shape]),
                       "shape for weight '%s' should be positive integer array" % name)
            init = item.get("init", "random")
            dtype = item.get("type", "float32")
            to_learn = item.get("update", True)
            self.make_weight(name, shape, init, dtype, to_learn)

        # layers
        for item in d.get("layer", []):
            name = item.pop("name", "")
            self.check_name(name)
            ltype = item.pop("type")
            if not ltype:
                self.error("missing type field in layer '%s'" % name)
            self.make_layer(name, ltype, item)

        # losses
        for item in d.get("loss", []):
            name = item.pop("name", "")
            self.check_name(name)
            ltype = item.pop("type", None)
            if not ltype:
                self.error("missing type field in loss '%s'" % name)
            self.make_loss(name, ltype, item)

        # training
        item = d.get("training", None)
        if not item:
            self.error("missing training section")
        if "loss" not in item:
            self.error("missing loss function in training section")
        optimizing_target = self.eval(item["loss"])
        self.check(isinstance(optimizing_target.father, Loss),
                   "the optimizing target should be a Loss instance")
        self.core.set_optimizing_target(optimizing_target)
        self.make_update(
            item.pop("method", "sgd"),
            item.pop("learning_rate", 0.1),
            item
        )
        output_infos = item.get("outputs", [])
        if not isinstance(output_infos, list):
            output_infos = [output_infos]
        for output_info in output_infos:
            output = self.eval(output_info)
            self.check(isinstance(output, NNValue),
                       "the output value '%s' should be a NNValue instance")
            self.core.add_output(output)

    def eval(self, v):
        if isinstance(v, str):
            if v.startswith('"') and v.endswith('"'):
                return v[1:-1]
            elif v in self.values:
                return self.values[v]
            else:
                names = v.strip().split(".")
                layer = self.core.eval(names[0])
                if not layer:
                    self.error("undefined layer '%s'" % names[0])
                sub = None if len(names) == 1 else names[1]
                value = layer.get_value(name=sub)
                self.check(isinstance(value, NNValue),
                           "%s.get_value() should return NNValue object for '%s'" % (layer.__class__.__name__, v))
                return value
        elif isinstance(v, int):
            return v
        elif isinstance(v, float):
            return v
        elif isinstance(v, list):
            return [self.eval(_) for _ in v]
        else:
            self.error("unknown value '%s' with type '%s' for eval()" % (v, v.__class__.__name__))

    def make_input(self, name, shape, dtype, data):
        if dtype not in self.valid_input_dtypes:
            self.error("invalid datatype '%s' for input '%s'" % (dtype, name))
        constructor = Constructor.get_constructor("input")
        if not constructor:
            self.error("input layer constructor missed")
        input_layer = constructor(name, shape, dtype, data)
        self.core.add_input(name, input_layer)

    def make_weight(self, name, shape, init_method, dtype, to_learn):
        if dtype not in self.valid_weight_dtypes:
            self.error("invalid datatype '%s' for weight '%s'" % (dtype, name))
        constructor = Constructor.get_constructor("weight")
        if not constructor:
            self.error("weight layer constructor missed")
        weight = constructor(name, shape, dtype, init_method)
        self.core.add_weight(name, weight, to_learn)

    def make_layer(self, name, ltype, item):
        constructor = Constructor.get_constructor(ltype)
        if not constructor:
            self.error("unknown layer type '%s' for layer '%s'" % (ltype, name))

        params = dict([(k, self.eval(item[k])) for k in item])
        layer = constructor(name, params, self.core)
        self.check(isinstance(layer, Layer),
                   "layer '%s: %s' should be Layer instance" % (name, ltype))
        self.core.add_layer(name, layer)

    def make_loss(self, name, ltype, item):
        constructor = Constructor.get_constructor(ltype)
        if not constructor:
            self.error("unknown loss type '%s' for loss '%s'" % (ltype, name))

        params = dict([(k, self.eval(item[k])) for k in item])
        loss = constructor(name, params, self.core)
        self.check(isinstance(loss, Loss),
                   "loss '%s: %s' should be Loss instance" % (name, ltype))
        self.core.add_loss(name, loss)

    def make_update(self, method, learning_rate, item):
        constructor = Constructor.get_constructor(method)
        if not constructor:
            self.error("unknown update type '%s'" % method)
        updater = constructor(learning_rate, item, self.core)
        self.check(isinstance(updater, Updater), "'%s' update should be Updater instance" % method)
        self.core.set_updater(updater)

    def check_name(self, s):
        if (not isinstance(s, str)) or s == "":
            self.error("invalid name '%s'" % s)
        elif s in self.names:
            self.error("duplicate definition '%s'" % s)
        elif s.find(".") >= 0:
            self.error("invalid name %s, '.' is used for internal names" % s)
        else:
            self.names.add(s)

    def backend_build(self):
        self.core.check_validity()
        builder = TheanoBackendBuilder(self.core)
        network = builder.build()
        self.check(isinstance(network, Network), "backend builder failed")
        return network
