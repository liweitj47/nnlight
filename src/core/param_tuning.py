from utility.debug import NNDebug


class ParameterTuningManager:

    def __init__(self, processor, comparator):
        self.processor = processor
        self.comparator = comparator

    def error(self, s):
        NNDebug.error("[" + self.__class__.__name__ + "] " + str(s))

    def check(self, cond, s):
        NNDebug.check(cond, "[" + self.__class__.__name__ + "] " + str(s))

    def tune(self, networks, params, topk=-1):
        """
        parameter tuning process
        :param networks: (config, Network obj) pair list
        :param params: additional parameter dict used by processor
        :param topk: if >=0, only top k results are returned
        :return: (config, network, processor result) triple ranked by comparator,
        """
        self.error("tune() method must be override")


class SimpleParameterTuningManager(ParameterTuningManager):

    def __init__(self, processor, comparator):
        ParameterTuningManager.__init__(self, processor, comparator)

    def tune(self, networks, params, topk=-1):
        results = []
        for config, network in networks:
            item = config, network, self.processor(network, params)
            results.append(item)
        results.sort(cmp=self.comparator)
        return results[:topk] if topk >= 0 else results
