# -*- coding: utf-8 -*-
"""

"""
from utility.debug import NNDebug


class Network:

    def __init__(self, core):
        self.core = core

    @staticmethod
    def error(s):
        NNDebug.error("[network] " + str(s))

    @staticmethod
    def check(cond, s):
        NNDebug.check(cond, "[network] " + str(s))

    def train(self):
        pass

    def test(self):
        pass

    def set_data(self, data_dict):
        pass

    def get_data(self, name):
        pass

    def save(self, path, weights=None):
        pass

    def load(self, path, weights=None):
        pass


