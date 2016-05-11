#encoding=utf-8
import sys
sys.path.append("../")
import numpy
import re
import MySQLdb
import make_wordvec
from core import  word2vec



'''print "loading data from MySQL..."
db = MySQLdb.connect(host="localhost",
                     user="bxq",
                     passwd="mengtaigu7",
                     charset="utf8",
                     db="xiecheng")
cursor = db.cursor()
cursor.execute("SELECT distinct q_seg FROM xiecheng.online_q ORDER BY q_id")
questions = [_[0].encode("utf-8") for _ in cursor.fetchall()]
'''

questions = []
preline = None
for line in open("../qq/input.txt"):
    '''if line.startswith("*****"):
        q = preline.strip().split("\t")[1]
        questions.append(q)
        print q
    preline = line'''
    if not line=="\n" and not line.startswith("***"):
        q = line.strip().split("\t")[1]
        questions.append(q) 
        


# word2vec
print "loading word2vec..."
w2v = word2vec.read("/home/bxq/nn/conv/data/vectors_azhang.bin")



outputf = open("test_questions.txt", "w")
for q in questions:
    outputf.write(q + "\n")

wordvec, mask = make_wordvec.convert(questions, w2v, 64)
numpy.save("test_salient.wordvec", wordvec)
numpy.save("test_salient.mask", mask)





