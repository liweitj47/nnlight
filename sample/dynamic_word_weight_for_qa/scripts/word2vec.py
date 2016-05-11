import numpy
import numpy
import re
import sys


def read(pathname):
    w2v = {}
    with open(pathname) as f:
        for line in f:
            segs = line.strip().split(" ")
            word = segs[0]
            vec = map(float, segs[1:])
            w2v[word] = numpy.asarray(vec, dtype="float32")
    return w2v


def convert(sentences, dictionary, max_len=64, extra_dim=0, transpose=True):
    data_size = len(sentences)
    wordvec_size = dictionary["</s>"].shape[0]
    if transpose:
        wordvec = numpy.zeros([data_size, wordvec_size+extra_dim, max_len], dtype="float32")
    else:
        wordvec = numpy.zeros([data_size, max_len, wordvec_size+extra_dim], dtype="float32")
    mask = numpy.zeros([data_size, max_len], dtype="float32")
    token_pattern = re.compile("([^ ]+?)/([a-zA-Z]+)")
    for i in range(data_size):
        for j, m in enumerate(token_pattern.finditer(sentences[i])):
            if j >= max_len:
                break
            word, pos = m.group(1), m.group(2)
            if word in dictionary:
                vec = dictionary[word]
                if transpose:
                    for k in range(wordvec_size):
                        wordvec[i][k][j] = vec[k]
                else:
                    wordvec[i][j] = vec
                mask[i][j] = 1.
    return wordvec, mask


if __name__ == "__main__":
    if len(sys.argv) == 5:
        sentences = open(sys.argv[1]).readlines()
        word2vec = read(sys.argv[2])
        wordvec, mask = convert(sentences, word2vec, int(sys.argv[4]))
        numpy.save(sys.argv[3] + ".wordvec", wordvec)
        numpy.save(sys.argv[3] + ".mask", mask)
    else:
        pass

