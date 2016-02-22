# -*- coding: utf-8 -*-
"""

"""
import numpy
import theano
from theano import tensor
from core.backend import BackendBuilder
from theano_network import TheanoNetwork
from utility.debug import NNDebug
from value.values import NNValue


class TheanoBackendBuilder(BackendBuilder):

    def __init__(self, core):
        BackendBuilder.__init__(self, core)
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
        for inp in self.core.inputs.values():
            data, shape = inp.get_data(), inp.get_shape()
            if not isinstance(data, numpy.ndarray):
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
        for inp in self.core.inputs.values():
            value = inp.get_value()
            given = diagram.get(value)
            repl = diagram.get_shared(value, self.maximum_sample_size)[i: j]
            theano_givens[given] = repl
        for inp in self.core.weights.values():
            value = inp.get_value()
            given = diagram.get(value)
            repl = diagram.get_shared(value, self.maximum_sample_size)
            theano_givens[given] = repl

        updater = self.core.updater
        if not hasattr(updater, "get_theano_updates"):
            self.error("missing get_theano_updates()' method for %s instance to "
                       "support Theano" % updater.__class__.__name__)
        theano_updates = updater.get_theano_updates(diagram, self.core)

        theano_outputs = []
        for output in self.core.output_target.values():
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
            outputs=theano_outputs,
            on_unused_input='ignore'  # some inputs is unnecessary since loss may not be computed
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
                              % (layer.__class__.__name__, layer.name))
            else:
                # for single output, an theano variable is returned
                # for multiple outputs, a NNValue->theano variable dictionary is returned
                output = layer.get_theano_output(self)
                if isinstance(output, dict):
                    NNDebug.check(value in output,
                                  "[builder] incomplete outputs for implementation of"
                                  " %s.get_theano_output()" % value.get_father().__class__.__name__)
                    for v in output:
                        self.mapping[v] = output[v]
                else:
                    self.mapping[value] = output
                result = self.mapping[value]
                NNDebug.check(isinstance(result, object),
                              "[builder] invalid outputs for implementation of"
                              " %s.get_theano_output()" % value.get_father().__class__.__name__)
                return result

    def get_shared(self, value, maximum_sample_size=None):
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
                              (layer.__class__.__name__, layer.name))
            else:
                shared = layer.get_theano_shared(maximum_sample_size=maximum_sample_size)
                self.shared_mapping[value] = shared
                return shared
