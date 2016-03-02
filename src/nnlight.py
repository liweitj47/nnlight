from core.builder import NNBuilder
from core.environment import NNEnvironment
from utility.configparser import ConfigParser


def create(src, data_dict, backend="theano", multi_configuration=False, environments=None):
    """
    build the network object from configuration source and
        assign data to the network

    :param src:
        if str, the configuration file path
        if dict, configuration python dictionary

    :param data_dict:
        dict, the key is the network input name defined by
        the configuration, and the value is the ndarray fed
        to the network's input

    :param backend:
        specifying the backend building system, default "theano"

    :param multi_configuration
        specifying whether enable multi-configuration support for  hyper-parameter
        tuning, if True, ={PARAM+} notation can be used in config file. And an array
        of (config, Network object) pairs with multiple configuration is returned

    :param environments
        dict, global parameters which can be retrieved by NNEnvironment.get()

    :return: the Network object(s)
    """
    if environments:
        for k in environments:
            NNEnvironment.set(k, environments[k])

    if multi_configuration and isinstance(src, str):
        networks = []
        parser = ConfigParser()
        configs = parser.parse(src, expand=True)
        for config in configs:
            builder = NNBuilder(backend)
            network = builder.build(config, data_dict)
            networks.append((config, network))
            return networks
    else:
        builder = NNBuilder(backend)
        return builder.build(src, data_dict)
