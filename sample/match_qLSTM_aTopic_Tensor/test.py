import sys
sys.path.append("../../src")
import math
import nnlight
import numpy
import random
import re



token_pattern = re.compile("([^ ]+?)/([a-zA-Z]+)")


pairs = [
            map(lambda sentence: [ (m.group(1), m.group(2))
                    for m in token_pattern.finditer(sentence) ],
                line.split("\t")
            )
            for line in open("../data/testdata.text").readlines()
        ] 



train_prefix = "../data/"
question_data = numpy.load(train_prefix + "testdata.wordvec.npy")
mask_data = numpy.load(train_prefix + "testdata.wordmask.npy")
answer_data = numpy.load(train_prefix + "testdata.atopic.npy")

span = numpy.load(train_prefix + "testdata.span.npy")

data_dict = {
    "questions":  question_data,
    "questions_mask": mask_data,
    "answers":  answer_data,
    "answers_neg": answer_data
}

group_size = span.shape[0]
test_size = question_data.shape[0]
batch_size = 200
batch_num = test_size / batch_size + 1

print "[loading data]"
for name in data_dict:
    print "%s: %s" % (name, data_dict[name].shape)
print


print "[testing]"
network = nnlight.create("match_qLSTM_aTopic_Tensor.config", data_dict)
network.load("qLSTM_aTopic_Tensor.model")
result = network.test(beg=0,
                      end=test_size,
                      batch=batch_size)[0]

p1 = 0.
p5 = 0.
mrr = 0.

print "[counting]"
for i in range(group_size):
    offset = span[i][0]
    pos_num = span[i][1]
    neg_num = span[i][2]

    if pos_num + neg_num == 0:
        continue

    scores = list(enumerate([x for x in result[offset:offset+pos_num+neg_num]]))
    scores = sorted(scores, key=lambda x:-x[1])[:10]

    if scores[0][0] >= pos_num and pos_num>0:
        print str(i) + " error!"
        print str.join("", [word for (word,pos) in pairs[offset+scores[0][0]][0]])
        print str.join("", [word for (word,pos) in pairs[offset+scores[0][0]][1]])
        print str.join("", [word for (word,pos) in pairs[offset+scores[0][0]][2]])
        print str.join("", [word for (word,pos) in pairs[offset][2]])
        print

    for j in range(len(scores)):
        if scores[j][0] < pos_num:
            if j==0:
                p1 += 1
            p5 += 1 if j<5 else 0
            mrr += 1.0 / (j+1)
            break

print p1/group_size, " ", p5/group_size, " ", mrr/group_size
            
    
