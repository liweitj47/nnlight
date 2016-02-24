import itertools
from layer.layers import Layer
from value.values import NNValue


class ValueInfo:
    def __init__(self, shape, dtype, value):
        self.shape = shape
        self.dtype = dtype
        self.value = value


class SmartLayer(Layer):

    def __init__(self, name, params, core):
        Layer.__init__(self, name, params, core)
        self.name = name
        self.params = params

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
                   "%s.info() should return a list of (name, usage, shape, [dtype]) tuple" % self.__class__.__name__)
        for info in infos:
            if len(info) == 3:
                info = (info[0], info[1], info[2], "")
            name, usage, template_shape, flags = info
            flags = flags.split(";")
            available_dtype = {"float32", "float64", "int32", "int64", "float", "int"}
            dtype = "float32"
            for d in flags:
                if d in available_dtype:
                    dtype = d
                    break
            extra = any([_ == "extra" for _ in flags])
            initial_shape = [x if isinstance(x, int) else -1 for x in template_shape]
            self.check(isinstance(template_shape, list) and
                       all([isinstance(x, str) or isinstance(x, int) for x in template_shape]),
                       "invalid shape for param '%s' defined in info()" % name)
            self.check(name not in self.all_info,
                       "duplicate info item '%s' defined in info()" % name)

            value = params.get(name)
            self.check(usage != "output" or value is None,
                       "'%s' already defined as output in info()" % name)
            if value is None:
                from utility.constructor import Constructor
                if usage == "weight":
                    constructor = Constructor.get_constructor("weight")
                    if not constructor:
                        self.error("weight layer constructor missed")
                    registry_name = self.__class__.__name__ + "." + name
                    weight = constructor(registry_name, initial_shape, dtype, "random")
                    core.add_weight(registry_name, weight)
                elif usage == "output":
                    value = Constructor.create_value(self, initial_shape, dtype)
                elif extra:
                    pass
                else:
                    self.error("missing input param '%s' defined in info() for '%s'" % (name, self.name))
            self.check(isinstance(value, NNValue) or extra,
                       "invalid input param '%s', expect a NNValue instance "
                       "defined in info() but actually receive a '%s'" % (name, value.__class__.__name__))

            info_obj = ValueInfo(template_shape, dtype, value)
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

        if len(self.inputs_info) < 1:
            self.error("at least one input should be defined in info()")
        if len(self.outputs_info) < 1:
            self.error("at least one output should be defined in info()")

        # add object attributes
        for name in self.all_info:
            if not hasattr(self, name):
                setattr(self, name, self.all_info[name].value)

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

    def check_input_type(self):
        pass

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
        for name, info in info_dict.items():
            if not info.value:
                continue  # extra parameter that is not offered
            shape = info.value.get_shape()
            shape_template = info.shape
            if len(shape) != len(shape_template):
                self.error("inconsistent dimension for '%s' 's %s '%s', %d expected "
                           "but actually %d" % (self.name, usage, name, len(shape_template), len(shape)))
            for i, (si, ti) in enumerate(zip(shape, shape_template)):
                if isinstance(ti, str):
                    res = self.try_parse_expression(ti, pool)
                    if res > 0:
                        ti = res
                    elif ti in pool:
                        ti = pool[ti]
                    elif ti in self.params:
                        self.check(isinstance(self.params[ti], int),
                                   "'%s' 's %dth element '%s' defined by info() expected "
                                   "to be an integer" % (name, i, ti))
                        ti = self.params[ti]
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

    def try_parse_expression(self, s, pool):
        s = s.strip()
        terms = []
        ops = []
        idx = 0

        def read_term(i):
            term = ""
            while i < len(s) and s[i] == ' ':
                i += 1
            while i < len(s) and s[i] not in {' ', '-', '+'}:
                term += s[i]
                i += 1
            if len(term) == 0:
                self.error("invalid format for shape element '%s' defined in info()" % s)
            return i, term

        def read_op(i):
            while i < len(s):
                if s[i] != ' ':
                    break
                i += 1
            else:
                return i, None
            if s[i] not in {'+', '-'}:
                self.error("invalid format for shape element '%s' defined in info()" % s)
            return i + 1, s[i]

        def eval_term(term):
            try:
                result = int(term)
            except ValueError:
                if not (term in pool or term in self.params):
                    self.error("unknown shape element '%s' of '%s' defined in info()" % (term, s))
                result = pool[term] if term in pool else self.params[term]
            self.check(isinstance(result, int),
                       "'%s' expected to be integer defined in info()" % term)
            return result

        idx, first_term = read_term(idx)
        terms.append(first_term)
        while idx < len(s):
            idx, next_op = read_op(idx)
            if next_op is None:
                break
            idx, next_term = read_term(idx)
            terms.append(next_term)
            ops.append(next_op)
        if len(terms) < 2:
            return -1
        else:
            res = eval_term(terms[0])
            for next_term, next_op in zip(terms[1:], ops):
                if next_op == '-':
                    res -= eval_term(next_term)
                elif next_op == '+':
                    res += eval_term(next_term)
            return res
