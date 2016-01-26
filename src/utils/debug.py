# -*- coding: utf-8 -*-
"""

"""
import random
import string
import theano
import numpy
import math


class NNDebug:
    """
    debug utility class
    """
    def __init__(self):
        pass

    @staticmethod
    def error(msg):
        raise Exception(msg)

    @staticmethod
    def warning(text):
        print text

    @staticmethod
    def check(cond, msg):
        if not cond:
            raise Exception(msg)
