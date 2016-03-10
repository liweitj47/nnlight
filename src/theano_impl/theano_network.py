# -*- coding: utf-8 -*-
"""

"""
import math
import numpy
from core.network import Network
from layer.basic.input import InputLayer
from layer.basic.weight import WeightLayer


class TheanoNetwork(Network):

    def __init__(self, core, base_train_func, base_test_func, maximum_sample_size):
        Network.__init__(self, core)
        self.inputs = self.core.inputs.values()
        self.base_train_func = base_train_func
        self.base_test_func = base_test_func
        self.total_sample_size = self.inputs[0].get_data().shape[0]
        self.maximum_sample_size =\
            maximum_sample_size \
            if 0 < maximum_sample_size < self.total_sample_size \
            else self.total_sample_size

    @staticmethod
    def __default_batch_collector(accu, v):
        if isinstance(accu, numpy.ndarray):
            if accu.shape == ():
                return numpy.array(accu + v)
            else:
                return numpy.concatenate([accu, v])
        else:
            return None

    def test(self, batch_size=None, batch_collector=None):
        if batch_size is None:
            batch_size = self.maximum_sample_size
        if batch_collector is None:
            batch_collector = self.__default_batch_collector

        result = None
        block_num = int(math.ceil(self.total_sample_size / float(self.maximum_sample_size)))

        for b in range(block_num):
            batch_num = 0
            block_beg = b * self.maximum_sample_size
            block_end = block_beg + self.maximum_sample_size
            if block_num > 1:
                for inp in self.inputs:
                    block = inp.get_data()[block_beg, block_end]
                    inp.set_theano_shared(block)
                    batch_num = int(math.ceil(block.shape[0] / float(batch_size)))

            for batch in range(batch_num):
                batch_begin = batch * batch_size
                batch_end = (batch+1) * batch_size
                if batch_end > block.shape[0]:
                    batch_end = block.shape[0]
                new_result = self.base_test_func(batch_begin, batch_end)

                if result is None:
                    result = new_result
                else:
                    result = [batch_collector(a, b) for a, b in zip(result, new_result)]

        return result

    def train(self, iters=None, batch_size=None, iter_reporter=None, batch_collector=None):
        if iters is None:
            iters = 1
        if batch_size is None:
            batch_size = self.maximum_sample_size
        if batch_collector is None:
            batch_collector = self.__default_batch_collector

        result = None
        block_num = int(math.ceil(self.total_sample_size / float(self.maximum_sample_size)))

        for i in range(iters):
            result = None
            for b in range(block_num):
                if block_num > 1:
                    block_beg = b * self.maximum_sample_size
                    block_end = min(block_beg + self.maximum_sample_size, self.total_sample_size)
                    for inp in self.inputs:
                        block = inp.get_data()[block_beg: block_end]
                        inp.set_theano_shared(block)
                else:
                    block_beg = 0
                    block_end = self.total_sample_size

                batch_num = int(math.ceil((block_end - block_beg) / float(batch_size)))
                for batch in range(batch_num):
                    batch_begin = batch * batch_size
                    batch_end = (batch+1) * batch_size
                    if batch_end > block_end:
                        batch_end = block_end
                    new_result = self.base_train_func(batch_begin, batch_end)

                    if result is None:
                        result = new_result
                    else:
                        result = [batch_collector(a, b) for a, b in zip(result, new_result)]

            if iter_reporter is not None:
                try:
                    iter_reporter(i, result)
                except Exception as e:
                    self.error("exception in iter_reporter '%s'" % e)

        return result

    def get_data(self, name):
        node = self.core.layers.get(name, None)
        if node is None:
            self.error("undefined input or weight layer '%s'" % name)
        elif not (isinstance(node, InputLayer) or isinstance(node, WeightLayer)):
            self.error("'%s' is not an input or weight layer" % name)
        else:
            return node.get_data()

    def set_data(self, data_dict):
        if not (isinstance(data_dict, dict)) and all([isinstance(x, numpy.ndarray) for x in data_dict.values()]):
            self.error("data dictionary passed to TheanoNetwork.set_data() should be a dict "
                       "with numpy.ndarray as values")
        for name in data_dict:
            node = self.core.layers.get(name, None)
            if node is None:
                self.error("undefined input or weight layer '%s'" % name)
            elif not (isinstance(node, InputLayer) or isinstance(node, WeightLayer)):
                self.error("'%s' is not an input or weight layer" % name)
        modified_layers = set()
        for name in data_dict:
            node = self.core.layers[name]
            node.set_data(data_dict[name])
            modified_layers.add(node)
        self.core.check_validity(modified_layers=modified_layers)
        self.maximum_sample_size = self.core.estimate_maximum_sample_size()
        # update shared variables
        for name in data_dict:
            node = self.core.layers[name]
            if isinstance(node, InputLayer):
                node.set_theano_shared(data_dict[name][:self.maximum_sample_size])
            else:
                node.set_theano_shared(data_dict[name])

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
        data_dict = None
        try:
            input_file = open(path, "rb")
            try:
                data_dict = numpy.load(input_file).item()
                if not isinstance(data_dict, dict):
                    raise IOError()
            except IOError:
                self.error("bad format file '%s'" % path)
            finally:
                input_file.close()
        except IOError:
            self.error("'%s' not found" % path)

        if isinstance(weights, list):
            weights = dict([(n, n) for n in weights])
        else:
            weights = dict([(n, n) for n in self.core.weights])

        if not (isinstance(data_dict, dict)) and all([isinstance(x, numpy.ndarray) for x in data_dict.values()]):
            self.error("invalid data dictionary for TheanoNetwork.load(), should be a dict "
                       "with numpy.ndarray as values")

        for name, bias in weights.items():
            if bias not in data_dict:
                self.error("missing weight data '%s'" % weights[bias])
            elif name not in self.core.weights:
                self.error("undefined weight data '%s'" % name)

        data_dict = dict([(n, data_dict[weights[n]]) for n in weights])
        self.set_data(data_dict)

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
        if isinstance(weights, list):
            weights = dict([(n, n) for n in weights])
        else:
            weights = dict([(n, n) for n in self.core.weights])

        data_dict = {}
        for name, bias in weights.items():
            if name not in self.core.weights:
                self.error("undefined weight data '%s'" % name)
            data_dict[bias] = self.get_data(name)

        try:
            output_file = open(path, "wb")
            numpy.save(output_file, data_dict)
            output_file.close()
        except IOError:
            self.error("save to '%s' failed" % path)
