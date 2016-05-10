#encoding=utf-8
import sys
sys.path.append("../")
import numpy
import re
import MySQLdb
import make_wordvec
from core import  word2vec


def convert(qa, buf):

    f = open(root + "tongyi/hit_tongyici_cilin.txt")
    hit_dict = {}
    wordsets = [ set(line.strip().split(" ")[1:]) for line in f ]
    for wordset in wordsets:
        for word in wordset:
            hit_dict[word] = wordset


    w2v = word2vec.read("/home/bxq/nn/conv/data/vectors_azhang.bin")

    token_pattern = re.compile("([^ ].*?)/([a-zA-Z]+)")


    # word configurations
    stop_pos = frozenset(["b", "e", "cc", "f", "mq", "qv", "u","g","w", "d", "c", "o", "y", "m", "ns", "nr", "nt"])
    stop_word = frozenset(["没", "无", "quot", "是", "有", "你", "您", "你们", "他", "我", "她", "它", "他们", "我们", "和", "而", "请", "在", "可以", "到", "个", "于", "为了", "好", "要", "去", "出", "来", "上", "天", "后", "gt", "多", "需", "为", "各", "会", "从", "里", "这", "这里", "lt", "amp", "如", "话", "说", "以", "与", "以及", "及", "大", "小", "人", "能", "没有", "点", "可", "本" ])


    for k, (q, a) in enumerate(qa):
    
        q_seg = [(m.group(1), m.group(2)) for m in token_pattern.finditer(q)]
        a_seg = [(m.group(1), m.group(2)) for m in token_pattern.finditer(a)]
    

        for i, (word, pos) in enumerate(q_seg):

            if pos in stop_pos or word in stop_word:
                continue

            fid = -1
        
            # strict occurance
            count = 0
            start = 0
            offset = len(word)
            while True:
                start = item[1].encode("utf-8").find(word, start)
                if start < 0:
                    break
                else:
                    start += offset
                    count += 1
            buf[k][fid][i] = count
            fid -= 1

            # tongyici
            count = 0
            simset = hit_dict.get(word, None)
            if simset is not None:
                for aword, apos in a_seg:
                    try:
                        if len(aword.decode("utf-8"))>1 and aword in simset:
                            count += 1
                    except Exception as e:
                        pass
            buf[k][fid][i] = count
            fid -= 1
        





