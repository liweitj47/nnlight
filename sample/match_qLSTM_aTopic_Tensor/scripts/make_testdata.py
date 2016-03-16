import re
import random
import word2vec as w2v
import numpy
import itertools


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
with open("../data/raw_data.txt") as f:
    for line in f:
        x = [_.strip() for _ in line.split("\t")]
        pair = Item(x[0], x[1], x[2]), Item(x[3], x[4], x[5])
        qapairs.append(pair)
qid_map = {}
for i, (q, a) in enumerate(qapairs):
    qid_map[q.idx] = i


print "building NS clusters..."
ns_pattern = re.compile(u"([^ ]+)/ns")
ns_map = {}
ns_full_map = {}
for q, a in qapairs:
    ns_set = set([m.group(1) for m in ns_pattern.finditer(q.seg)])
    for ns in ns_set:
        ns_map.setdefault(ns, set()).add(q.idx)
    ns_full_map.setdefault(",".join(ns_set), set()).add(q.idx)
    

print "loading querys..."
queries = {}
with open("../data/raw_query.txt") as f:
    for line in f:
        x = line.strip().split("\t")
        item = Item(x[0], x[1], x[2])
        if item.idx in qid_map:
            queries.setdefault(item, []).append(x[-1])


print "building candidate sets..."
testset = []
for item in queries:
    ns_set = set([m.group(1) for m in ns_pattern.finditer(item.seg)])
    candidates = ns_full_map.get(",".join(ns_set), set())
    pos_ids = set(queries[item])
    neg_ids = candidates.difference(pos_ids)
    if len(neg_ids) == 0:
        for ns in ns_set:
            candidates = candidates.union(ns_map.get(ns, set()))
        neg_ids = candidates.difference(pos_ids)
    pos_ids = pos_ids.intersection(candidates)
    neg_ids = random.sample(neg_ids, min(15, len(neg_ids)))
    if len(pos_ids) > 0:
        testset.append( (item, pos_ids, neg_ids) )
    print str(item.idx) + "\t" + item.raw
    print "************************************************************"
    for idx in pos_ids:
        print str(idx) + "\t" + qapairs[qid_map[idx]][0].raw 
    print "************************************************************"
    for idx in neg_ids:
        print str(idx) + "\t" + qapairs[qid_map[idx]][0].raw 
    print


print "laoding answer representations"
data_size = sum([len(pos)+len(neg) for (_, pos, neg) in testset])
all_answers = numpy.load("../data/all_answers.topic")
answers = numpy.zeros([data_size, all_answers.shape[1]], dtype="float32")
spans = numpy.zeros([len(queries), 3], dtype="int")
questions = [None for _ in range(data_size)]
texts = [None for _ in range(data_size)]


print "loading wordvec..."
word2vec = w2v.read("../data/word_vectors.dat")


print "building numpy matrix..."
offset = 0
for i, (item, pos_ids, neg_ids) in enumerate(testset):
    # span
    spans[i] = offset, len(pos_ids), len(neg_ids)
    # answer
    for k in itertools.chain(pos_ids, neg_ids):
        match = qapairs[qid_map[k]]
        questions[offset] = item.seg
        answers[offset] = all_answers[qid_map[k]]
        texts[offset] = item.raw + "\t" + match[0].raw + "\t" + match[1].raw
        offset += 1
wordvec, mask = w2v.convert(questions, word2vec, max_len=64, extra_dim=0)



print "saving numpy matrix..."
numpy.save(open("../data/testdata.question","w"), wordvec)
numpy.save(open("../data/testdata.mask", "w"), mask)
numpy.save(open("../data/testdata.answer", "w"), answers)
numpy.save(open("../data/testdata.span", "w"), spans)
with open("../data/testdata.text", "w") as f:
    for line in texts:
        f.write(line + "\n")
