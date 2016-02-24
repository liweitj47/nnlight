import itertools
from layer.smart_layer import SmartLayer


class ValueInfo:
    def __init__(self, shape, dtype, value):
        self.shape = shape
        self.dtype = dtype
        self.value = value


class TheanoSmartLayer(SmartLayer):

    # is it necessary?
    def get_theano_output(self, diagram):
        class SmartTheanoAgent:
            def __init__(self): pass
        agent = SmartTheanoAgent()
        for name, info in itertools.chain(self.inputs_info.items(), self.weights_info.items()):
            var = diagram.get(info.value) if info.value else None
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
                   "or get_theano_output() in your smart layer %s" % self.__class__.__name__)
