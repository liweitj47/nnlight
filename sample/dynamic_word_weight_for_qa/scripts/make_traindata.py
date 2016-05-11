#encoding=utf-8
import sys
sys.path.append("../")
import numpy
import re
import word2vec


print "loading HIT tongyici..."
hit_dict = {}
with open("../data/hit_tongyici_cilin.txt") as f:
    for line in f:
        wordset = set(line.strip().split(" ")[1:])
        for word in wordset:
            hit_dict[word] = wordset



print "loading QA data..."
items = []
with open("../data/raw_data.txt") as f:
    for line in f:
        segs = line.strip().split("\t")
        if len(segs) == 4:
            items.append(segs)



# word2vec
print "loading word2vec..."
w2v = word2vec.read("../data/word_vectors.txt")




# word configurations
stop_pos = frozenset(["b", "e", "cc", "f", "mq", "qv", "u","g","w", "d", "c", "o", "y", "m", "ns", "nr", "nt"])
stop_word = frozenset(["没", "无", "quot", "是", "有", "你", "您", "你们", "他", "我", "她", "它", "他们", "我们", "和", "而", "请", "在", "可以", "到", "个", "于", "为了", "好", "要", "去", "出", "来", "上", "天", "后", "gt", "多", "需", "为", "各", "会", "从", "里", "这", "这里", "lt", "amp", "如", "话", "说", "以", "与", "以及", "及", "大", "小", "人", "能", "没有", "点", "可", "本" ])



# extract
print "start extracting..."
data_size = len(items)
max_len = 64
token_pattern = re.compile("([^ ].*?)/([a-zA-Z]+)")
salience = numpy.zeros([data_size, max_len], dtype="float32")
questions = []
qid = 0
outfile = open("../data/raw_questions.txt", "w")

for q, q_seg, a, a_seg in items:    
    a_seg = [(m.group(1), m.group(2)) for m in token_pattern.finditer(a_seg)]

    usable = False
    output_string = ""

    for i, m in enumerate(token_pattern.finditer(q_seg)):
        
        if i >= max_len:
            break

        word, pos = m.group(1), m.group(2)
        if pos in stop_pos or word in stop_word:
            continue

        count = 0
        
        # strict occurance
        start = 0
        offset = len(word)
        while True:
            start = a.find(word, start)
            if start < 0:
                break
            else:
                start += offset
                count += 1

        # tongyici
        simset = hit_dict.get(word, None)
        if simset is not None:
            for aword, apos in a_seg:
                if len(aword)>1 and aword in simset:
                    count += 1
        
        if count > 1:
            usable = True
            salience[qid][i] = 1.0
            output_string += "【" + word + "】"
        else:
            output_string += word

    if usable > 0:
        qid += 1
        questions.append(q_seg)
        outfile.write(output_string + "\n")

        

print "saving numpy data..."
vec, mask = word2vec.convert(questions, w2v, max_len, 0, transpose=False)
numpy.save(open("../data/salience.wordvec", "w"), vec)
numpy.save(open("../data/salience.mask", "w"), mask)
numpy.save(open("../data/salience.salience", "w"), salience[:qid])





