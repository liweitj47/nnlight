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
from layer.dropoutable import Dropoutable


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
        base_train_func, base_test_func = self.build_functions()
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

    def build_givens(self, i, j, diagram):
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
        return theano_givens

    def build_outputs(self, diagram):
        theano_outputs = []
        for output in self.core.output_target:
            theano_outputs.append(diagram.get(output))
        return theano_outputs

    def build_train_func(self, i, j):
        diagram = TheanoDiagram(dropout=True)
        givens = self.build_givens(i, j, diagram)
        outputs = self.build_outputs(diagram)
        updater = self.core.updater
        if not hasattr(updater, "get_theano_updates"):
            self.error("missing get_theano_updates()' method for %s instance to "
                       "support Theano" % updater.__class__.__name__)
        updates = updater.get_theano_updates(diagram, self.core, i, j)
        train_func = theano.function(
            inputs=[i, j],
            givens=givens,
            updates=updates,
            outputs=outputs
        )
        return train_func, (diagram, givens, outputs)

    def build_test_func(self, i, j, cur):
        if cur is None:
            diagram = TheanoDiagram()
            givens = self.build_givens(i, j, diagram)
            outputs = self.build_outputs(diagram)
        else:
            diagram, givens, outputs = cur
        return theano.function(
            inputs=[i, j],
            givens=givens,
            outputs=outputs,
            on_unused_input='ignore'  # some inputs is unnecessary since loss may not be computed
        )

    def should_build_separately(self):
        for layer in self.core.layers.values():
            if isinstance(layer, Dropoutable) and layer.dropout_rate() > 0:
                return True
        return False

    def build_functions(self):
        i, j = tensor.lscalar(), tensor.lscalar()
        train_func, byproducts = self.build_train_func(i, j)
        if self.should_build_separately():
            byproducts = None
        test_func = self.build_test_func(i, j, byproducts)
        return train_func, test_func


class TheanoDiagram:

    def __init__(self, dropout=False):
        self.mapping = {}
        self.shared_mapping = {}
        self.dropout = dropout

    def wrap(self, layer, value):
        output = layer.get_theano_output(self)
        if not isinstance(output, dict):
            output = {value: output}
        else:
            NNDebug.check(value in output,
                          "[builder] incomplete outputs for implementation of"
                          " %s.get_theano_output()" % value.get_father().__class__.__name__)

        for v in output:
            o = output[v]
            if False and self.dropout and isinstance(layer, Dropoutable) and layer.can_dropout(v):
                numpy_rng = numpy.random.RandomState(233)
                from theano.tensor.shared_randomstreams import RandomStreams
                theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
                o = o * theano_rng.binomial(size=v.get_shape(),
                                        n=1, p=1-layer.dropout_rate(),
                                        dtype=theano.config.floatX)
            self.mapping[v] = o

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
                self.wrap(layer, value)
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
