from __future__ import print_function
import sys
try:
    reload(sys)
    sys.setdefaultencoding('utf-8')
except:
    pass

import codecs
import csv
import os
os.getcwd()
os.chdir(r'/mnt/data/input/TextRank')
from textrank4zh import TextRank4Keyword, TextRank4Sentence
#TFIDF提取Keywords实现
import jieba.posseg as pseg
import os

class TFIDF:
    def __init__(self):
        self.idf_file = 'IDF.txt'
        self.idf_dict, self.common_idf = self.load_idf()

    def build_wordsdict(self, text):
        word_dict = {}
        candi_words = []
        candi_dict = {}
        for word in pseg.cut(text):
            if word.flag[0] in ['n', 'v', 'a'] and len(word.word) > 1:
                candi_words.append(word.word)
            if word.word not in word_dict:
                word_dict[word.word] = 1
            else:
                word_dict[word.word] += 1
        count_total = sum(word_dict.values())
        for word, word_count in word_dict.items():
            if word in candi_words:
                candi_dict[word] = word_count/count_total
            else:
                continue

        return candi_dict

    def extract_keywords(self, text, num_keywords):
        keywords_dict = {}
        candi_dict = self.build_wordsdict(text)
        for word, word_tf in candi_dict.items():
            word_idf = self.idf_dict.get(word, self.common_idf)
            word_tfidf = word_idf * word_tf
            keywords_dict[word] = word_tfidf
        keywords_dict = sorted(keywords_dict.items(), key=lambda asd:asd[1], reverse=True)

        return keywords_dict[:num_keywords]

    def load_idf(self):
        idf_dict = {}
        for line in open(self.idf_file,encoding='utf-8'):
            word, freq = line.strip().split(' ')
            idf_dict[word] = float(freq)
        common_idf = sum(idf_dict.values())/len(idf_dict)

        return idf_dict, common_idf
        
    
    csv_file=open('ai_challenger_test_1000.csv',encoding='utf-8')    #打开csv文件
csv_reader_lines = csv.reader(csv_file) #逐行读取csv文件
date=[]    #创建列表准备接收csv各行数据

renshu = 0
output=open('data_textrank_xx.txt','w',encoding='utf-8')
for one_line in csv_reader_lines:
    date.append(one_line)    #将读取的csv分行数据按行存入列表‘date’中
    renshu = renshu + 1
#取出每一行转化为text，顺带判断有无\ufeffbios，一并去掉
#output=open('D:\\machine study\\NLP\\doc_topic_class\\output\\data_textrank.txt','w',encoding='utf-8')
for i in range(len(date)):
    list1=[]
    list2=[]
    list3=[]
    result=[]
    result1=[]
    b=c=0
    a=''
    c=i
    output.write(str(c))
    output.write('\t')
    output.write('关键词以及权重分别为：')
    text=str(date[i]).replace('[\'','').replace('\']','').replace('\\ufeffbios','')
#    print()
#    print('第'+str(i)+'行的textrank')
#    print()
    tr4w = TextRank4Keyword()

    tr4w.analyze(text=text, lower=True, window=2)   # py2中text必须是utf8编码的str或者unicode对象，py3中必须是utf8编码的bytes或者str对象
    for item in tr4w.get_keywords(10, word_min_len=2):        
          list1.append([item.word,item.weight])
         
    tfidfer = TFIDF()
    for keyword in tfidfer.extract_keywords(text, 10):
       list2.append(list(keyword))  
    for i in range(len(list1)):
        for j in range(len(list2)):  
           if list1[i][0]==list2[j][0]:
                 result.append(list1[i])
#        if list1[i][0]!=list2[j][0]:
#            result1.append([i,list1[i]])
    for i in list1:
        if i not in result:
            result1.append(i)         
    length=len(result)

    if length<5:
        if len(result1)>=5-length:           
           for i in range(5-length):
               result.append(result1[i])
    
    for item in result:
        a=item[0]
        b=item[1]            
        output.write('\t')
        output.write(str(a))
        output.write('\t')
        output.write(str(b))
    output.write('\t')

    tr4s = TextRank4Sentence()
    tr4s.analyze(text=text, lower=True, source = 'all_filters')    
 
    #print( '第'+str(i)+'行的'+'摘要：' )
    output.write('摘要为：')
    for item in tr4s.get_key_sentences(num=3):
        d=item.index
        e=item.weight
        f=item.sentence
        output.write('\t')
        output.write(str(d))
        output.write('\t')
        output.write(str(e))
        output.write('\t')
        output.write(str(f))     
    output.write('\t')
    output.write('\n')           
output.close()        
    