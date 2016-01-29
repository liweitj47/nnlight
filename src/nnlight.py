from builder import NNBuilder


def create(src, data_dict):
    builder = NNBuilder()
    return builder.build(src, data_dict)

