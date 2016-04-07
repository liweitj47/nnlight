class Dropoutable:

    def __init__(self, dropout):
        self.dropout = dropout
        self.undropoutable_values = set()

    def dropout_rate(self):
        return self.dropout

    def can_dropout(self, v):
        return v not in self.undropoutable_values

    def mark_undroptable(self, v):
        self.undropoutable_values.add(v)
