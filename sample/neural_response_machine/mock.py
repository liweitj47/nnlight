import numpy
import nnlight
import theano
theano.config.profile = True


data_dict = {
    "source": numpy.zeros([200, 32, 100], dtype="float32"),
    "source_mask": numpy.ones([200, 32], dtype="float32"),
    "target": numpy.zeros([200, 32, 100], dtype="float32"),
    "target_mask": numpy.ones([200, 32], dtype="float32")
}

network = nnlight.create("mock.config", data_dict)
print "start..."
print network.train(batch_size=200)[0].shape

