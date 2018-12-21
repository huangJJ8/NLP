from __future__ import print_function
import mlstudiosdk.modules
import mlstudiosdk.solution_gallery
from mlstudiosdk.modules.components.component import LComponent
from mlstudiosdk.modules.components.settings import Setting
from mlstudiosdk.modules.components.utils.orange_table_2_data_frame import table2df
from mlstudiosdk.modules.components.utils.orange_table_2_data_frame import df2table
from mlstudiosdk.modules.algo.data import Domain, Table
from mlstudiosdk.modules.algo.evaluation import Results
from mlstudiosdk.modules.algo.data.variable import DiscreteVariable, ContinuousVariable
from mlstudiosdk.modules.utils.itemlist import MetricFrame
from mlstudiosdk.modules.utils.metricType import MetricType
from sklearn.model_selection import train_test_split
from mlstudiosdk.exceptions.exception_base import Error
from TextRank4Keyword import TextRank4Keyword
from TF_IDF import TF_IDF
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import itertools
import tempfile
import random
import string
import shutil
import jieba
import jieba.posseg as pseg
import glob  # find file directories and files
import codecs
import csv
import re
import os
import warnings
warnings.filterwarnings('ignore')
import sys
try:
    reload(sys)
    sys.setdefaultencoding('utf-8')
except:
    pass



class Text_Extract_Keywords(LComponent):
    category = 'Nature Language Processing'
    name = "Text Extract Keywords"
    title = "Text Extract Keywords"
  
    inputs = [("Train Data", mlstudiosdk.modules.algo.data.Table, "set_traindata")
              ]
    
    outputs = [
        ("News", mlstudiosdk.modules.algo.data.Table),
        ("Predictions", Table),
        ("Evaluation Results", mlstudiosdk.modules.algo.evaluation.Results),
        ("Columns", list),
        ("Metas", list),
        ("Metric Score", MetricFrame),
        ("Metric", MetricType)
    ]
    n_keywords = Setting(5, {"type": "integer", "minimum": 0, "exclusiveMinimum": True})


    def __init__(self):
        super().__init__()
        self.train_data = None
        self.keywords_num=5
          
    def set_traindata(self, data):
        self.train_data = data

    def get_keywords_num(self):
        # get keywords_num 
        #key_dict = Setting(5, {"type": "number", "enum": [1,2, 3, 4, 5,6],"minimum": 0, "exclusiveMinimum": True})
        #self.keywords_num=key_dict.default
        self.keywords_num = self.n_keywords
        

    def run(self):
        # text = codecs.open('test.csv', 'r', 'utf-8').read()
        self.get_keywords_num()      
        train_data=table2df(self.train_data)
        if train_data.index.size == 0:
            raise AttributesError_new("Missing data, the input dataset should have two rows and one column "
                              "(the first row is column name,  the second row is the single text.)")
        elif train_data.columns.size > 1:
            raise AttributesError_new("Input data should have only one column")
        elif train_data.index.size > 1:
            raise AttributesError_new("The single text should be filled into one row")
        """ use 1 col and 1 row data """
        train_data = str(train_data.values[0][0])
        #use Textrank to get the keywords
        tr4w = TextRank4Keyword()
        tr4w.analyze(text=train_data, lower=True, window=5)   
        list1=[]
        list2=[]
        list3=[]
        list4=[]
        list5=[]
        list6=[]
        result=[]
        for item in tr4w.get_keywords(self.keywords_num*2, word_min_len=2):
            list1.append([item.word, item.weight])
            list6.append(item.word)
        self.keywords_set=list(set(list6))
#         print('keywords_set:',self.keywords_set)         
#         print(list1)
        
        #use TF_IDF to get the keywords
        tfidfer = TF_IDF()
        word_dict,candi_dict=tfidfer.build_wordsdict(train_data)
        word_dict1=tfidfer.build_wordsdict1(train_data)
        for keyword in tfidfer.extract_keywords(train_data, self.keywords_num*2):
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
        if length>=self.keywords_num:
            for i in range(self.keywords_num):
                result.append(list3[i])
        else:
            result=list3
            if len(list4)>=self.keywords_num-length:           
                for i in range(self.keywords_num-length):
                    list3.append(list4[i])
                    result=list3
        result.sort(key= lambda k:k[1],reverse=True)
        for i in range(len(result)):
            for word, word_tf in word_dict1.items():
                if result[i][0]==word:
                    result[i].append(int(word_tf))
        # print('result1：',result)  
        
        for i in range(len(result)):
            list7 = []
            list7.append(result[i][0])
#             print('list7:',list7)
            mapping=list(map(lambda x: self.keywords_set.index(x),list7))
#             print('mapping:',mapping)
            result[i][0]=mapping[0]
        # print('result2：',result) 

        metas =[DiscreteVariable('keywords',self.keywords_set),
                ContinuousVariable('weight'),
                 ContinuousVariable('word_frequency')]
#         print('Domain(metas):',Domain(metas))
#         listma=[[1,2,3],[4,5,6],[3,5,6]]
#         print(listma)
        domain=Domain(metas)
#         print('domain.attributes:',domain.attributes)
#         print('domain.class_vars:',domain.class_vars)
        final_result=Table.from_list(domain,result)
#         final_result=Table.from_list(Domain(metas),listma)
        print('final_result:',final_result)
       
        self.send('News', final_result)
        self.send("Metas", metas)