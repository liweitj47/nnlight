import sys
sys.path.append("../../src/")
import nnlight
import math
import numpy
import re
import random


question_data = numpy.load("data/salience.wordvec")
mask_data = numpy.load("data/salience.mask")
salience_data = numpy.load("data/salience.salience")

train_size = question_data.shape[0]
batch_size = 200
iter_num = 300
data_dict = {
    "sentence":  question_data[:train_size],
    "mask": mask_data[:train_size],
    "salience":  salience_data[:train_size]
}


print "[loading data]"
for name in data_dict:
    print "%s: %s" % (name, data_dict[name].shape)
print


print "[building network]"
network = nnlight.create("salience_lstm.config", data_dict)

print "training size: %d" % train_size
print "batch size: %d" % batch_size
print "iterations: %d" % iter_num
print


def reporter(i, result):
    total_likelihood = result[0]
    print result[1]
    avg_likelihood = math.exp(-total_likelihood * batch_size / train_size)
    if i%5==4:
        network.save("salience_lstm.model")
    print "iters %s: likelihood %f" % (i, avg_likelihood)


print network.test(batch_size=200)[1]
print "[training]"
#network.load("salience_lstm.model")
network.train(batch_size=batch_size,
              iters=iter_num,
              iter_reporter=reporter)


