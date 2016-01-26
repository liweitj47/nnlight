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
        weights = [diagram.get(w.get_output()) for w in core.to_learn.values()]
        updates = []
        for weight in weights:
            grad = theano.grad(loss, weight)
            updates.append((
                weight,
                weight - self.learning_rate * grad
            ))
        return updates
