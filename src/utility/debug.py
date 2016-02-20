# -*- coding: utf-8 -*-
"""

"""


class NNDebug:
    """
    debug utility class
    """
    def __init__(self):
        pass

    @staticmethod
    def error(msg):
        raise NNException(msg)

    @staticmethod
    def warning(text):
        print text

    @staticmethod
    def check(cond, msg):
        if not cond:
            raise NNException(msg)


class NNException(Exception):
    pass
