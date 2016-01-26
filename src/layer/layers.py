# -*- coding: utf-8 -*-
"""

"""
from utils.debug import NNDebug


class Layer:

    def __init__(self):
        self.name = type(self) # default name
        pass

    def error(self, s):
        NNDebug.error("[" + type(self) + "] " + str(s))

    def check(self, cond, s):
        NNDebug.check(cond, "[" + type(self) + "] " + str(s))

    def program_check(self, cond, s):
        NNDebug.check(cond, "[" + type(self) + "] " + str(s))

    def get_inputs(self):
        pass

    def get_outputs(self):
        pass

    def get_value(self, name=None):
        """
        retrieve the NNValue bound to the layer
        :param name: reference string for some NNValue of the layer, by default None
        :return: the NNValue bound to the layer, by default the output value
        """
        pass

    def check_input_type(self):
        pass

    def forward_shape(self, override=False):
        pass

    def backward_shape(self, override=False):
        pass


class LayerWithData(Layer):

    def get_data(self):
        pass

    def set_data(self, data):
        pass

    def get_shape(self):
        pass






