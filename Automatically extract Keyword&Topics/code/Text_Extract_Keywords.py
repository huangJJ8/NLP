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
from TextRank4Keyword import TextRank4Keyword
#TFIDF提取Keywords实现
import jieba.posseg as pseg
from TF_IDF import TF_IDF


class Text_Extract_Keywords():
    def __init__(self):
        self.train_data = None
        
    def run(self,keywords_num=5):
# text = codecs.open('test.csv', 'r', 'utf-8').read()
        tr4w = TextRank4Keyword()
        tr4w.analyze(text=self.train_data, lower=True, window=2)   
        list1=[]
        list2=[]
        list3=[]
        list4=[]
        result=[]
#         print( '关键词：' )
        for item in tr4w.get_keywords(keywords_num+2, word_min_len=2):
            list1.append([item.word, item.weight])
#         print(list1)
        tfidfer = TF_IDF()
        for keyword in tfidfer.extract_keywords(self.train_data, keywords_num+2):
            list2.append(list(keyword))  
#         print(list2)
        for i in range(len(list1)):
            for j in range(len(list2)):  
                if list1[i][0] == list2[j][0]:
                    list3.append(list1[i])
                    
        for i in list1:
            if i not in list3:
                list4.append(i)         
                    
        length=len(list3)
#         print(list3)
#         print(list4)
        if length>=keywords_num:
            for i in range(keywords_num):
                result.append(list3[i])
        else:
            result=list3
            if len(list4)>=keywords_num-length:           
                for i in range(keywords_num-length):
                    list3.append(list4[i])
                    result=list3
        result.sort(key= lambda k:k[1],reverse=True)
        print(result)
        return result
    
if __name__ == '__main__':
    s = Text_Extract_Keywords()
    s.train_data = codecs.open('test.csv', 'r', 'utf-8').read()
    s.run(7) 