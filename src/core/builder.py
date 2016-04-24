# -*- coding: utf-8 -*-
"""

"""
import re
from core import NNCore
from layer.layers import Layer
from loss.loss import Loss
from network import Network
from backend import BackendBuilder
from theano_impl.theano_builder import TheanoBackendBuilder
from computation_on_java_impl.computation_on_java_builder import ComputationOnJavaBackendBuilder
from updater.updaters import Updater
from utility.configparser import ConfigParser
from utility.constructor import Constructor
from utility.debug import NNDebug, NNException
from value.values import NNValue


class NNBuilder:

    def __init__(self, backend="theano"):
        # supporting datatypes
        self.valid_input_dtypes = {"float32", "int64"}
        self.valid_weight_dtypes = {"float32"}

        # support name pattern
        self.name_pattern = re.compile("[a-zA-Z_][a-zA-Z0-9_]*")

        # internal states
        self.core = NNCore()
        self.names = set()
        self.groups = {}
        self.values = {}

        # backend builder
        if backend == "theano":
            self.backend_builder = TheanoBackendBuilder(self.core)
        elif backend == "computation_on_java":
            self.backend_builder = ComputationOnJavaBackendBuilder(self.core)
        else:
            try:
                constructor = Constructor.load_constructor_from_path(backend)
                self.check(isinstance(constructor, BackendBuilder),
                           "invalid backend builder '%s'" % backend)
                self.backend_builder = constructor(self.core)
            except NNException:
                self.error("invalid backend builder '%s'" % backend)

        # constructor map reassign
        cm = self.backend_builder.get_constructor_map()
        if cm is not None:
            self.check(isinstance(cm, dict),
                       "%s.get_constructor_map() should return a "
                       "Constructor:Alias dict" % self.backend_builder.__class__.__name__)
            Constructor.load_constructor_map(cm)

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
                the configuration, and the value is the ndarray
                fed to the network's input

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

        # group
        for item in d.get("group", []):
            name = item.get("name", "")
            self.check_name(name)
            item.pop("name")
            group = {}
            weight_defs = set()
            for key in item:
                val = item[key]
                if isinstance(val, list) or isinstance(val, int):
                    d.setdefault("weight", []).append({"name": key, "shape": val})
                    weight_defs.add(key)
                else:
                    group[key] = val
                    if val in weight_defs:
                        weight_defs.remove(key)
            for key in weight_defs:
                group[key] = key
            self.groups[name] = group

        # inputs
        for item in d.get("input", []):
            name = item.get("name", "")
            self.check_name(name)
            self.expand_group(item)
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

        # weight
        for item in d.get("weight", []):
            name = item.get("name", "")
            self.check_name(name)
            self.expand_group(item)
            shape = self.eval(item.get("shape"))
            if shape is None:
                self.error("missing shape field for weight '%s'" % name)
            if isinstance(shape, int):
                shape = [-1 for _ in range(shape)]
            self.check(isinstance(shape, list) and all([isinstance(x, int) for x in shape]),
                       "shape for weight '%s' should be an integer array" % name)
            init = item.get("init", "random")
            dtype = item.get("type", "float32")
            to_learn = item.get("update", True)
            self.make_weight(name, shape, init, dtype, to_learn)

        # layers
        for item in d.get("layer", []):
            name = item.pop("name", "")
            self.check_name(name)
            self.expand_group(item)
            self.make_layer(name, item)

        # losses
        for item in d.get("loss", []):
            name = item.pop("name", "")
            self.check_name(name)
            self.expand_group(item)
            self.make_loss(name, item)

        # training
        item = d.get("training", None)
        if not item:
            self.error("missing training section")
        trainable = "loss" in item
        if trainable:
            optimizing_target = self.eval(item["loss"])
            self.check(isinstance(optimizing_target.father, Loss),
                       "the optimizing target should be a Loss instance")
            self.core.set_optimizing_target(optimizing_target)
            self.make_update(item)
        output_infos = item.get("outputs", [])
        if not isinstance(output_infos, list):
            output_infos = [output_infos]
        if len(output_infos) == 0 and not trainable:
            self.error("at least one output should be given in 'outputs' field")
        for output_info in output_infos:
            output = self.eval(output_info)
            self.check(isinstance(output, NNValue),
                       "the output value '%s' should be a NNValue instance")
            self.core.add_output(output_info, output)

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
                           "%s.get_value() should return NNValue object for '%s', "
                           "maybe '%s' is not defined." % (layer.__class__.__name__, v, v))
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
        input_layer = constructor(name, {"shape": shape, "dtype": dtype, "data": data}, self.core)
        self.core.add_input(name, input_layer)

    def make_weight(self, name, shape, init_method, dtype, to_learn):
        if dtype not in self.valid_weight_dtypes:
            self.error("invalid datatype '%s' for weight '%s'" % (dtype, name))
        constructor = Constructor.get_constructor("weight")
        if not constructor:
            self.error("weight layer constructor missed")
        weight = constructor(name, {"shape": shape, "dtype": dtype, "init_method": init_method}, self.core)
        self.core.add_weight(name, weight, to_learn)

    def get_constructor(self, typ, classpath, source):
        if classpath:
            constructor = Constructor.load_constructor_from_path(classpath)
            if typ:
                Constructor.register_type(typ, classpath, constructor)
        elif typ:
            constructor = Constructor.get_constructor(typ)
        else:
            constructor = None
            self.error("missing type field for %s" % source)
        if not constructor:
            self.error("unknown type '%s' for %s" % (typ, source))
        return constructor

    def make_layer(self, name, item):
        classpath = item.pop("classpath", None)
        ltype = item.pop("type", None)
        constructor = self.get_constructor(ltype, classpath, "layer '%s'" % name)
        params = dict([(k, self.eval(item[k])) for k in item])
        layer = constructor(name, params, self.core)
        self.check(isinstance(layer, Layer),
                   "layer '%s: %s' should be Layer instance" % (name, ltype))
        self.core.add_layer(name, layer)

    def make_loss(self, name, item):
        ltype = item.pop("type", None)
        classpath = item.pop("classpath", None)
        constructor = self.get_constructor(ltype, classpath, "loss '%s'" % name)
        params = dict([(k, self.eval(item[k])) for k in item])
        loss = constructor(name, params, self.core)
        self.check(isinstance(loss, Loss),
                   "loss '%s: %s' should be Loss instance" % (name, ltype))
        self.core.add_loss(name, loss)

    def make_update(self, item):
        method = item.pop("method", "sgd")
        classpath = item.pop("classpath", None)
        constructor = self.get_constructor(method, classpath, "parameter updater")
        learning_rate = item.pop("learning_rate", float(0.1)),
        updater = constructor(learning_rate, item, self.core)
        self.check(isinstance(updater, Updater), "'%s' update should be Updater instance" % method)
        self.core.set_updater(updater)

    def check_name(self, s):
        if (not isinstance(s, str)) or s == "":
            self.error("invalid name '%s'" % s)
        elif not self.name_pattern.match(s):
            self.error("invalid name '%s'" % s)
        elif s in self.names:
            self.error("duplicate definition '%s'" % s)
        elif s.find(".") >= 0:
            self.error("invalid name %s, '.' is used for internal names" % s)
        else:
            self.names.add(s)

    def expand_group(self, item):
        gname = item.get("group")
        if gname in self.groups:
            group = self.groups[gname]
            for name in group:
                if name in item:
                    self.error("duplicate parameter '%s' of group '%s', already"
                               " exists in '%s'" % (name, gname, item["name"]))
                item[name] = group[name]
            item.pop("group")

    def backend_build(self):
        self.core.check_validity()
        network = self.backend_builder.build()
        self.check(isinstance(network, Network), "backend builder failed")
        return network
