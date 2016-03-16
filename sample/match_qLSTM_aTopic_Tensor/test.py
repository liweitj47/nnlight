import sys
sys.path.append("../../src")
import math
import nnlight
import numpy
import random


print "[loading data]"
prefix = "data/testdata"
question = numpy.load(prefix + ".question")
mask = numpy.load(prefix + ".mask")
answer = numpy.load(prefix + ".answer")
span = numpy.load(prefix + ".span")
data_dict = {
    "questions":  question,
    "questions_mask": mask,
    "answers":  answer
}
for name in data_dict:
    print "%s: %s" % (name, data_dict[name].shape)
print


print "[predicting scores]"
network = nnlight.create("test.config", data_dict)
network.load("qLSTM_aTopic_Tensor.model")
result = network.test(batch_size=200)[0]


print "[evaluating results]"
groups = span.shape[0]
p1 = 0.
p5 = 0.
for i in range(groups):
    offset, pos, neg = span[i]
    if pos + neg == 0:
        continue
    scores = sorted([_ for _ in enumerate(result[offset:offset+pos+neg])],
                    key=lambda _:-_[1])
    for j, score in enumerate(scores):
        if score[0] < pos:
            if j == 0:
                p1 += 1
            if j < 5:
                p5 += 1
            break
print "P@1:", p1/groups, ", P@5:", p5/groups
