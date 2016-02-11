# -*- coding: utf-8 -*-
"""

"""
from utility.debug import NNDebug


class Layer:

    def __init__(self, name, params, core):
        self.name = name  # default name

    def error(self, s):
        NNDebug.error("[" + self.__class__.__name__ + "] " + str(s))

    def check(self, cond, s):
        NNDebug.check(cond, "[" + self.__class__.__name__ + "] " + str(s))

    def program_check(self, cond, s):
        NNDebug.check(cond, "[" + self.__class__.__name__ + "] " + str(s))

    '''
    def get_inputs(self):

    def get_outputs(self):

    def get_value(self, name=None):
        """
        retrieve the NNValue bound to the layer
        :param name: reference string for some NNValue of the layer, by default None
        :return: the NNValue bound to the layer, by default the output value
        """

    def check_input_type(self)

    def forward_shape(self, override=False)

    def backward_shape(self, override=False)
    '''


class LayerWithData(Layer):

    def get_data(self):
        pass

    def set_data(self, data):
        pass

    def get_shape(self):
        pass






