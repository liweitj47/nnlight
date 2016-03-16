import word2vec as w2v
import numpy
import re


class Item:
    def __init__(self, idx, raw, seg):
        self.idx = idx
        self.raw = raw
        self.seg = seg

    def __hash__(self):
        return self.idx.__hash__()

    def __eq__(self, other):
        return self.idx == other.idx


print "loading basic data..."
qapairs = []
qid_map = {}
with open("../data/raw_data.txt") as f:
    for line in f:
        x = [_.strip() for _ in line.split("\t")]
        pair = Item(x[0], x[1], x[2]), Item(x[3], x[4], x[5])
        qid_map[pair[0].idx] = len(qapairs)
        qapairs.append(pair)


print "loading answer representations..."
all_answers = numpy.load("../data/all_answers.topic")


print "extracting short questions..."
max_len = 64
token_pattern = re.compile(u"[^ ]+?/[a-zA-Z]+")
qapairs = filter(lambda (q, a): len(token_pattern.findall(q.seg))<max_len, qapairs)


print "building question representations..."
questions = [q.seg for (q, a) in qapairs]
word2vec = w2v.read("../data/word_vectors.dat")
vec, mask = w2v.convert(questions, word2vec, max_len, extra_dim=0)


print "building answer topic matrix..."
answers = numpy.zeros([len(questions), all_answers.shape[1]], dtype="float32")
for i, (q, a) in enumerate(qapairs):
    answers[i] = all_answers[qid_map[q.idx]]


print "saving qa texts..."
with open("../data/traindata.text","w") as f:
    for q, a in qapairs:
        f.write(q.raw + "\t" + a.raw + "\n")


print "saving numpy matrix..."
root = "../data/traindata"
numpy.save(open(root + ".question", "w"), vec)
numpy.save(open(root + ".mask", "w"), mask)
numpy.save(open(root + ".answer", "w"), answers)

