# -*- coding: utf-8 -*-
"""

"""
from utility.debug import NNDebug


class Network:

    def __init__(self, core):
        self.core = core

    def error(self, s):
        NNDebug.error("[" + self.__class__.__name__ + "] " + str(s))

    def check(self, cond, s):
        NNDebug.check(cond, "[" + self.__class__.__name__ + "] " + str(s))

    def train(self):
        self.error("train() method must be override")

    def test(self):
        self.error("test() method must be override")

    def set_data(self, data_dict):
        self.error("set_data() method must be override")

    def get_data(self, name):
        self.error("get_data() method must be override")

    def save(self, path, weights=None):
        self.error("save() method must be override")

    def load(self, path, weights=None):
        self.error("load() method must be override")
