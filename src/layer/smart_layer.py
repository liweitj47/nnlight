import itertools
from layers import Layer
from utils.constructor import Constructor
from value.values import NNValue


class ValueInfo:
    def __init__(self, shape, dtype, value):
        self.shape = shape
        self.dtype = dtype
        self.value = value


class SmartLayer(Layer):

    def __init__(self, name, params, core):
        Layer.__init__(name)
        self.name = name

        #  ValueInfo dict for layer's components
        self.inputs_info = {}
        self.weights_info = {}
        self.outputs_info = {}
        self.all_info = {}

        #  inputs and outputs NNValue
        self.inputs = []
        self.outputs = []

        #  read into value infos
        infos = self.info()
        self.check(isinstance(infos, list) and
                   all([(isinstance(x, tuple) or isinstance(x, list)) and len(x) >= 3 for x in infos]),
                   "%s.info() should return a list of (name, usage, shape, [dtype]) tuple" % type(self))
        for info in infos:
            if len(info) == 3:
                info = (info[0], info[1], info[2], "float32")
            name, usage, shape, dtype = info
            self.check(isinstance(shape, list) and all([isinstance(x, str) or isinstance(x, int) for x in shape]),
                       "invalid shape for param '%s' defined in info()" % name)
            self.check(name not in self.all_info,
                       "duplicate info item '%s' defined in info()" % name)

            value = params.get(name)
            if value is None:
                if usage == "weight":
                    constructor = Constructor.get_constructor("weight")
                    if not constructor:
                        self.error("weight layer constructor missed")
                    registry_name = str(type(self)) + "." + name
                    weight = constructor(registry_name, shape, dtype, "random")
                    core.add_weight(registry_name, weight)
                elif usage == "output":
                    value = Constructor.create_value(self, shape, dtype)
                else:
                    self.error("missing input param '%s' defined in info()" % name)
            self.check(isinstance(value, NNValue),
                       "invalid input param '%s', expect a NNValue instance "
                       "defined in info() but actually receive a '%s'" % (name, type(value)))

            info_obj = ValueInfo(shape, dtype, value)
            if usage == "input":
                self.all_info[name] = info_obj
                self.inputs_info[name] = info_obj
                self.inputs.append(value)
            elif usage == "output":
                self.all_info[name] = info_obj
                self.outputs_info[name] = info_obj
                self.outputs.append(value)
            elif usage == "weight":
                self.all_info[name] = info_obj
                self.weights_info[name] = info_obj
                self.inputs.append(value)  # weight is seen as an input
            else:
                self.error("invalid usage string for param '%s' defined in info(),"
                           " (input|output|weight)" % usage)

            # add object attributes
            for name in self.all_info:
                if not hasattr(self, name):
                    setattr(self, name, self.all_info[name])

    def info(self):
        self.error("you must override info() method for your smart layer")
        return []

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs

    def get_value(self, name=None):
        if name is None:
            return self.get_outputs()[0]
        else:
            return self.all_info.get(name, None)

    def forward_shape(self, override=False):
        pool = {}
        self.broadcast_shape("input", self.inputs_info, pool, False)
        self.broadcast_shape("weight", self.weights_info, pool, False)
        self.broadcast_shape("output", self.outputs_info, pool, override)

    def backward_shape(self, override=False):
        pool = {}
        self.broadcast_shape("output", self.outputs_info, pool, False)
        self.broadcast_shape("weight", self.weights_info, pool, False)
        self.broadcast_shape("input", self.inputs_info, pool, False)

    def broadcast_shape(self, usage, info_dict, pool, override=False):
        for name, info in info_dict:
            shape = info.value.get_shape()
            shape_template = info.shape
            if len(shape) != len(shape_template):
                self.error("inconsistent dimension for '%s' 's %s '%s', %d expected "
                           "but actually %d" % (self.name, usage, name, len(shape_template), len(shape)))
            for i, (si, ti) in enumerate(zip(shape, shape_template)):
                if isinstance(ti, str):
                    if ti in pool:
                        ti = pool[ti]
                    else:
                        if shape[i] > 0:
                            pool[ti] = si
                        continue
                if ti > 0:
                    if 0 < si != ti:
                        if override:
                            shape[i] = ti
                        else:
                            self.error("inconsistent shape for '%s' 's %s '%s', %dth element expected to "
                                       "be %d but actually %d" % (self.name, usage, name, i, ti, si))
                    elif si <= 0:
                        shape[i] = ti
            info.value.set_shape(shape)

    # is it necessary?
    def get_theano_output(self, diagram):
        class SmartTheanoAgent:
            def __init__(self): pass
        agent = SmartTheanoAgent()
        for name, info in itertools.chain(self.inputs_info.items(), self.weights_info.items()):
            var = diagram.get(info.value)
            setattr(agent, name, var)
        for name, info in self.outputs_info.items():
            setattr(agent, name, None)
        self.get_theano_output_smart(agent)
        result = {}
        for name, info in self.outputs_info.items():
            result[info.value] = getattr(agent, name)
        return result

    def get_theano_output_smart(self, n):
        self.error("you should either implement get_theano_output_smart() "
                   "or get_theano_output() in your smart layer %s" % type(self))
