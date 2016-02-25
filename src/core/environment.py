# naive implementation currently
env = {}


class NNEnvironment:

    def __init__(self):
        pass

    @staticmethod
    def set(k, v):
        env[k] = v

    @staticmethod
    def get(k):
        return env.get(k)
