import sys
sys.path.append("../../src")
import math
import nnlight
import numpy


train_prefix = "../data/cnn_traindata"
data_dict = {
    "questions": numpy.load(train_prefix + ".wordvec"),
    "questions_mask": numpy.load(train_prefix + ".wordmask"),
    "answers": numpy.load(train_prefix + ".atopic")
}

print "[loading data]"
for name in data_dict:
    print "%s: %s" % (name, data_dict[name].shape)
print


print "[building network]"
network = nnlight.create("match_qCnn_aTopic.config", data_dict)

train_size = data_dict["questions"].shape[0]
batch_size = 200
batch_num = train_size / batch_size + 1
iter_num = 40
print "training size: %d" % train_size
print "batch size: %d" % batch_size
print "iterations: %d" % iter_num
print


def reporter(i, result):
    total_likelihood = result[0]
    avg_likelihood = math.exp(-total_likelihood/batch_num)
    print "iters %s: likelihood %f" % (i, avg_likelihood)

print "[training]"
network.train(beg=0,
              end=train_size,
              batch=batch_size,
              iters=iter_num,
              iter_reporter=reporter)
print

print "[saving model]"
print "save to 'qCNN_aTopic.model'"
network.save("qCNN_aTopic.model")



