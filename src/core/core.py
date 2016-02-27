# -*- coding: utf-8 -*-
"""

"""
import math

from layer.basic.input import InputLayer
from layer.basic.weight import WeightLayer
from layer.layers import Layer
from loss.loss import Loss
from updater.updaters import Updater
from utility.debug import NNDebug
from value.values import NNValue, NNScalar


class NNCore:

    def __init__(self):
        #  mapping from name to the layer objects
        self.layers = {}
        self.weights = {}
        self.inputs = {}
        self.losses = {}

        #  dict for to-be-trained weights
        self.to_learn = {}

        #  dict of NNValues for output information
        self.output_target = []
        #  we want outputs be ordered so use another dict for names
        self.output_name_mapping = {}

        self.updater = None
        self.optimizing_target = None

        #  topology sorted list for layers
        self.tps_list = None

    @staticmethod
    def error(s):
        NNDebug.error("[builder] " + str(s))

    @staticmethod
    def check(cond, s):
        NNDebug.check(cond, "[builder] " + str(s))

    @staticmethod
    def program_check(cond, s):
        NNDebug.check(cond, "[bug in code] " + str(s))

    def eval(self, name):
        return self.layers.get(name, None)

    def add_layer(self, name, l):
        self.program_check(isinstance(l, Layer),
                           "NNCore.add_layer() need pass a Layer object")
        self.check(name not in self.layers,
                   "Layer '%s' already defined in NNCore" % name)
        self.layers[name] = l

    def add_weight(self, name, w, to_learn=True):
        self.program_check(isinstance(w, WeightLayer),
                           "NNCore.add_weight() need pass a WeightLayer object")
        self.check(name not in self.layers,
                   "Layer '%s' already defined in NNCore" % name)
        self.weights[name] = w
        self.layers[name] = w  # weight is a special layer
        if to_learn:
            self.to_learn[name] = w

    def add_input(self, name, i):
        self.program_check(isinstance(i, InputLayer),
                           "NNCore.add_input() need pass an InputLayer object")
        self.check(name not in self.layers,
                   "Layer '%s' already defined in NNCore" % name)
        self.inputs[name] = i
        self.layers[name] = i  # input is a special layer

    def add_loss(self, name, l):
        self.program_check(isinstance(l, Loss),
                           "NNCore.add_loss() need pass a Loss object")
        self.check(name not in self.layers,
                   "Layer '%s' already defined in NNCore" % name)
        self.losses[name] = l
        self.layers[name] = l  # loss is a special layer

    def add_output(self, name, obj):
        self.program_check(isinstance(obj, NNValue),
                           "NNCore.add_output() need pass a NNValue object")
        self.check(obj not in self.output_name_mapping,
                   "duplicate output info '%s', maybe"
                   " it's alias already exists" % name)
        self.output_target.append(obj)
        self.output_name_mapping[obj] = name

    def set_optimizing_target(self, t):
        self.program_check(isinstance(t, NNScalar),
                           "NNCore.set_optimizing_target() need pass a Loss object")
        self.optimizing_target = t

    def set_updater(self, upd):
        self.program_check(isinstance(upd, Updater),
                           "NNCore.set_updater() need pass a Updater object")
        self.updater = upd

    def check_validity(self, modified_layers=None):
        """
        check the structure validity and type safety for the network diagram, includes:
        1) input content validity
        2) shape consistency
        3) no structure cycling

        :param modified_layers: set type, if the method is called on a data changing operation
                                (set_data() for input or weight, it should point out which layers
                                are changed and shape broadcasting will begin from them.
        """
        if not modified_layers:
            origin_layers = set()

        self.topology_sort()
        self.check_inputs()
        self.forward_shape(modified_layers)
        self.backward_shape()
        self.check_unknown_shape()

    def forward_shape(self, origin_layers=None):
        if not origin_layers:
            origin_layers = set()
        visited_layers = {}

        def dfs(cur, override):
            if cur in visited_layers:
                return visited_layers[cur]
            else:
                has_override_child = False
                children = [i.father for i in cur.get_inputs()]
                for child in children:
                    # each child layer shape can be override only once
                    has_override_child = dfs(child,
                                             override and not has_override_child)
                if (cur in origin_layers) != has_override_child:
                    override &= True
                cur.check_input_type()
                cur.forward_shape(override)
                return visited_layers.setdefault(cur, override)

        global_override = True if len(origin_layers) > 0 else False
        for layer in reversed(self.tps_list):
            dfs(layer, global_override)

    def backward_shape(self):
        visited_layers = set()

        def dfs(cur):
            if cur not in visited_layers:
                cur.forward_shape()
                children = [i.father for i in cur.get_inputs()]
                for child in children:
                    dfs(child)
                visited_layers.add(cur)

        for layer in reversed(self.tps_list):
            dfs(layer)

    def topology_sort(self):
        roots = [_.father for _ in [self.optimizing_target] + self.output_target]
        result = []
        visited = set()  # not finished!
        finished = set()  # finished!

        def dfs(cur):
            if cur in finished:
                return
            elif cur in visited:
                self.error("cyclic dependency detected in the network at '%s'" % cur.name)
            else:
                visited.add(cur)
            for inp in cur.get_inputs():
                dfs(inp.father)
            finished.add(cur)
            visited.remove(cur)
            result.append(cur)

        for root in roots:
            dfs(root)
        self.tps_list = result

    def check_inputs(self):
        sample_size = None
        if len(self.inputs) == 0:
            self.error("missing input section for network")
        for name in self.inputs:
            inp = self.inputs[name]
            if inp not in self.tps_list:
                self.error("unused input '%s' detected" % inp.name)
            else:
                shape = inp.get_shape()
                if len(shape) == 0:
                    self.error("zero dimensional input data not allowed: '%s' " % inp.name)
                elif sample_size is not None and shape[0] != sample_size:
                    self.error("inconsistent input sample size for '%s', if your data's first dimension "
                               "does not stand for sample size, use weight section instead" % name)
                else:
                    sample_size = shape[0]
        for name in self.weights:  # weight can be seen as input
            weight = self.weights[name]
            if weight not in self.tps_list:
                self.error("unused weight '%s' detected" % weight.name)

    def check_unknown_shape(self):
        for name in self.layers:
            l = self.layers[name]
            values = l.get_inputs() + l.get_outputs()
            for v in values:
                shape = v.get_shape()
                self.check(all([x > 0 for x in shape]),
                           "cannot determine explicit shape for layer '%s': %s"
                           % (l.name, ['?' if d <= 0 else d for d in shape]))

    def estimate_maximum_sample_size(self):
        total_sample_size = self.inputs.values()[0].get_shape()[0]
        total_mem_size = 0
        maximum_mem_size = 4 * (1024 ** 3)
        value_set = set()
        for layer in self.layers.values():
            for v in layer.get_outputs():
                value_set.add(v)
            for v in layer.get_inputs():
                value_set.add(v)
        for value in value_set:
            shape = value.get_shape()
            elements_num = reduce(lambda x, y: x*y, shape, 1)
            total_mem_size += elements_num * value.get_element_size()
        if total_mem_size > maximum_mem_size * 0.9:
            scale = total_mem_size / (maximum_mem_size * 0.9)
            return int(math.floor(total_sample_size / scale))
        else:
            return total_sample_size
