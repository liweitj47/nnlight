import numpy
import nnlight
import theano
theano.config.profile = True


data_dict = {
    "question": numpy.zeros([10000, 32, 100], dtype="float32"),
    "question_mask": numpy.ones([10000, 32], dtype="float32"),
    "answer": numpy.zeros([10000, 32, 100], dtype="float32"),
    "answer_mask": numpy.ones([10000, 32], dtype="float32")
}

network = nnlight.create("local_match1.config", data_dict)
print network.test(batch_size=1000)[1]

