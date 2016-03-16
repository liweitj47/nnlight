import numpy
import random
import math
import sys
sys.path.append("../../src")
import nnlight


train_size = 180000
train_prefix = "data/traindata"
questions = numpy.load(train_prefix + ".question")[:train_size]
questions_mask = numpy.load(train_prefix + ".mask")[:train_size]
answers = numpy.load(train_prefix + ".answer")[:train_size]
answers_neg = numpy.zeros(answers.shape, dtype="float32")


data_dict = {
    "questions": questions,
    "questions_mask": questions_mask,
    "answers": answers,
    "answers_neg": answers_neg
}
print "[loading data]"
for name in data_dict:
    print "%s: %s" % (name, data_dict[name].shape)
print


print "[building network]"
network = nnlight.create("train.config", data_dict)

batch_size = 200
batch_num = int(math.ceil(train_size / float(batch_size)))
iter_num = 40
iter_index = 0
print "training size: %d" % train_size
print "batch size: %d" % batch_size
print "iterations: %d" % iter_num
print


def reporter(inner_iter, result):
    margin, bad_count = result[0], result[1]
    network.save("qLSTM_aTopic_Tensor.model")
    print "iters %s: margin %f, error_count %d" % (inner_iter + 10*iter_index, -margin/batch_num, int(bad_count))


print "[training]"
while iter_index < iter_num:
    for i in range(train_size):
        answers_neg[i] = answers[int(random.random()*train_size)]
    network.set_data({"answers_neg": answers_neg[:train_size]})
    network.train(
            batch_size=batch_size,
            iters=10,
            iter_reporter=reporter)
    iter_index += 1

print

print "[saving model]"
network.save("qLSTM_aTopic_Tensor.model")
