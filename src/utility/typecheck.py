
int_types = {"int32", "int64"}


class TypeChecker:

    def __init__(self):
        pass

    @staticmethod
    def consistent(tp1, tp2):
        if tp1 == tp2:
            return True
        elif tp1 in int_types and tp2 in int_types:
            return True
        else:
            return False
