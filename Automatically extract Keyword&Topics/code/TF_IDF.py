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
import jieba.posseg as pseg
import jieba

class TF_IDF():
    def __init__(self):
        self.idf_file = 'IDF.txt'
        self.idf_dict, self.common_idf = self.load_idf()

    def build_wordsdict(self, text):
        word_dict = {}
        candi_words = []
        candi_dict = {}
        for word in pseg.cut(text):
            if word.flag[0] in ['an', 'i', 'j', 'l', 'n', 'nr', 'nrfg', 'ns', 'nt', 'nz', 't', 'v', 'vd', 'vn', 'eng']  #['n', 'v', 'a']
            and len(word.word) > 1:
                candi_words.append(word.word)
            if word.word not in word_dict:
                word_dict[word.word] = 1
            else:
                word_dict[word.word] += 1
#         print('word_dict:',word_dict)
        count_total = sum(word_dict.values())
        for word, word_count in word_dict.items():
            if word in candi_words:
                candi_dict[word] = word_count/count_total
            else:
                continue

        return word_dict,candi_dict


    def build_wordsdict1(self, text):
        word_dict1 = {}
        candi_words = []
        candi_dict = {}
        text=text.lower()
        for word in jieba.cut(text):
            if len(word) > 1:
                candi_words.append(word)
            if word not in word_dict1:
                word_dict1[word] = 1
            else:
                word_dict1[word] += 1
        return word_dict1

    def extract_keywords(self, text, num_keywords):
        keywords_dict = {}
        word_dict,candi_dict = self.build_wordsdict(text)
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
    
    
if __name__ == '__main__':
    pass