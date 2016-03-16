# -*- coding: utf-8 -*-
"""

"""
import theano


class Updater:

    def __init__(self, learning_rate, params, core):
        pass


class SGDUpdater(Updater):

    def __init__(self, learning_rate, params, core):
        Updater.__init__(self, learning_rate, params, core)
        self.learning_rate = learning_rate

    def get_theano_updates(self, diagram, core):
        loss = diagram.get(core.optimizing_target)
        updates = []
        for weight in core.to_learn.values():
            weight_value = weight.get_value()
            weight_symbol = diagram.get(weight_value)
            weight_data = diagram.get_shared(weight_value)
            grad = theano.grad(loss, weight_symbol)  # real data is used through given clause thus not part of graph
            updates.append((
                weight_data,
                weight_data - self.learning_rate * grad
            ))
        return updates

