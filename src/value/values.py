# -*- coding: utf-8 -*-
"""

"""


class NNValue:

    def __init__(self, father):
        self.father = father

    def get_father(self):
        return self.father

    def get_element_size(self):
        pass

    def get_shape(self):
        pass

    def set_shape(self, shape):
        pass


class NNScalar(NNValue):

    def __init__(self, father):
        NNValue.__init__(self, father)

    def get_shape(self):
        return []


class NNScalarInt64(NNScalar):

    def get_element_size(self):
        return 8


class NNScalarFloat32(NNScalar):

    def get_element_size(self):
        return 4


class NNArray(NNValue):

    def __init__(self, shape, father):
        NNValue.__init__(self, father)
        self.shape = shape

    def get_shape(self):
        return self.shape

    def set_shape(self, shape):
        self.shape = shape


class NNArrayInt64(NNArray):

    def get_element_size(self):
        return 8


class NNArrayFloat32(NNArray):

    def get_element_size(self):
        return 4

