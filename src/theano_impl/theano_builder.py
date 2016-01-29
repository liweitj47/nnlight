# -*- coding: utf-8 -*-
"""

"""
import numpy
import theano

from utils.debug import NNDebug
from theano import tensor
from theano_network import TheanoNetwork
from value.values import NNValue


class TheanoBackendBuilder:

    def __init__(self, core):
        self.core = core
        self.maximum_sample_size = -1
        self.tps_list = []

    @staticmethod
    def error(s):
        NNDebug.error("[builder] " + str(s))

    @staticmethod
    def check(cond, s):
        NNDebug.check(cond, "[builder] " + str(s))

    def build(self):
        self.check_inputs()
        self.maximum_sample_size = self.core.estimate_maximum_sample_size()
        base_train_func, base_test_func = self.build_theano()
        network = TheanoNetwork(core=self.core,
                                base_train_func=base_train_func,
                                base_test_func=base_test_func,
                                maximum_sample_size=self.maximum_sample_size)
        return network

    def check_inputs(self):
        for _, inp in self.core.inputs:
            data, shape = inp.get_data(), inp.get_shape()
            if not isinstance(data(), numpy.ndarray):
                self.error("input data '%s' should be of numpy.ndarray instance" % inp.name)
            if len(data.shape) != len(shape) or \
               any([x > 0 and x != y for (x, y) in zip(shape, data.shape)]):
                self.error("inconsistent input shape for '%s', expect %s but %s actually"
                           % (inp.name, shape, data.shape))

    def build_theano(self):
        diagram = TheanoDiagram()
        diagram.get(self.core.optimizing_target)

        i, j = theano.tensor.lscalar(), theano.tensor.lscalar()

        theano_givens = {}
        for inp in self.core.inputs.values() + self.core.weights.values():
            given = diagram.get(inp.get_value())
            repl = diagram.get_shared(inp.get_value(), self.maximum_sample_size)
            theano_givens[given] = repl

        updater = self.core.updater
        if not hasattr(updater, "get_theano_updates"):
            self.error("missing get_theano_updates()' method for %s instance to support Theano" % type(updater))
        theano_updates = updater.get_theano_updates(diagram, self.core)

        theano_outputs = []
        for output in self.core.output_target:
            theano_outputs.append(diagram.get(output))

        theano_train_func = theano.function(
            inputs=[i, j],
            givens=theano_givens,
            updates=theano_updates,
            outputs=theano_outputs
        )
        theano_test_func = theano.function(
            inputs=[i, j],
            givens=theano_givens,
            outputs=theano_outputs
        )
        return theano_train_func, theano_test_func


class TheanoDiagram:

    def __init__(self):
        self.mapping = {}
        self.shared_mapping = {}

    def get(self, value):
        if not isinstance(value, NNValue):
            NNDebug.error("[builder] TheanoDiagram.get() acquire NNValue instance")

        if value in self.mapping:
            return self.mapping[value]
        else:
            layer = value.get_father()
            if layer is None:
                NNDebug.error("[builder] missing father field for NNValue passed to TheanoDiagram.get()")
            elif not hasattr(layer, "get_theano_output"):
                NNDebug.error("[builder] 'missing get_theano_output()' method for %s instance '%s' to support Theano"
                              % (type(layer), layer.name))
            else:
                # for single output, an theano variable is returned
                # for multiple outputs, a NNValue->theano variable dictionary is returned
                output = layer.get_theano_output(self)
                if isinstance(output, dict):
                    NNDebug.check(value in output,
                                  "[builder] incomplete outputs for implementation of"
                                  " %s.get_theano_output()" % type(value.get_father()))
                    for v in output:
                        self.mapping[v] = output[v]
                else:
                    self.mapping[value] = output
                result = self.mapping[value]
                NNDebug.check(isinstance(result, object),
                           "[builder] invalid outputs for implementation of"
                                  " %s.get_theano_output()" % type(value.get_father()))
                return result

    def get_shared(self, value, maximum_sample_size):
        if not isinstance(value, NNValue):
            NNDebug.error("[builder] TheanoDiagram.get_shared() acquire NNValue instance")

        if value in self.shared_mapping:
            return self.shared_mapping[value]
        else:
            layer = value.get_father()
            if layer is None:
                NNDebug.error("[builder] missing father field for NNValue passed to TheanoDiagram.get_shared()")
            elif not hasattr(layer, "get_theano_shared") or not hasattr(layer, "set_theano_shared"):
                NNDebug.error("[builder] 'missing get/set_theano_shared()' method "
                              "for %s instance '%s' to support Theano" %
                              (type(layer), layer.name))
            else:
                shared = layer.get_theano_shared(self, maximum_sample_size=maximum_sample_size)
                self.shared_mapping[value] = shared
                return shared