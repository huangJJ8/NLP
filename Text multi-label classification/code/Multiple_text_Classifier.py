# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 23:55:26 2018

@author: chenjing34
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import mlstudiosdk.modules
import mlstudiosdk.solution_gallery
from mlstudiosdk.modules.components.component import LComponent
from mlstudiosdk.modules.components.utils.orange_table_2_data_frame import table2df
from mlstudiosdk.modules.components.utils.orange_table_2_data_frame import df2table
from mlstudiosdk.modules.algo.data import Domain, Table
from mlstudiosdk.modules.algo.evaluation import Results
from mlstudiosdk.modules.algo.data.variable import DiscreteVariable, ContinuousVariable
from mlstudiosdk.modules.utils.itemlist import MetricFrame
from mlstudiosdk.modules.utils.metricType import MetricType
from sklearn.model_selection import train_test_split
from mlstudiosdk.exceptions.exception_base import Error
import itertools
import jieba
import os
import glob #find file directories and files
import fasttext  #model
import codecs
import re
import warnings
warnings.filterwarnings('ignore')


class Multiple_text_Classifier(LComponent):
    category = 'Algorithm'
    name = "Multiple_text_Classifier"
    
    inputs = [("Train Data", mlstudiosdk.modules.algo.data.Table, "set_traindata"),
              ("Test Data", mlstudiosdk.modules.algo.data.Table, "set_testdata")
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
    
    
    def __init__(self):
        super().__init__()
        self.train_data = None
        self.test_data = None

    
    def set_traindata(self, data):
        self.train_data = data

    def set_testdata(self, data):
        self.test_data = data

    def label_str2number(self,data):
        label_str_raw = list(set(data[self.label_name]))
        format = lambda x: int(label_str_raw.index(x))
        data[self.label_name] = data[self.label_name].map(format)
        return data,label_str_raw

    def label_number2str(self,data,label_str_raw):
        format = lambda x: label_str_raw[x]
        data = data.astype({self.label_name:'int'})
        data[self.label_name] = data[self.label_name].map(format)
        return data

    def merge_result(self,data_len,legal_pre, legal_pre_pro,legal_test_data_id):
        # combine pre and predict possibility that can be predicted and data can not(illegal data)
        pre = [len(self.label_domain)-1]*data_len
        pre_prob = [[None]*(len(self.label_domain)-1)]*data_len
        for id_index in range(len(legal_pre)):
            id = legal_test_data_id[id_index]
            pre[id] = legal_pre[id_index]
            pre_prob[id] = legal_pre_pro[id_index]
        return np.array(pre),np.array(pre_prob)


    def files(self,curr_dir = '.', ext = 'fasttext_*.txt'):
        """Files in the current directory"""
        for i in glob.glob(os.path.join(curr_dir, ext)):
            yield (i)

    def remove_files(self,rootdir, ext, show = False):
        """Delete the matching files in the rootdir directory"""
        for i in self.files(rootdir, ext):
             #if show:
                #print('如下文件已被删除:',i)
            os.remove(i)
            
    def get_name(self):
        # get label & sentence columns name
        self.label_name = self.train_data.domain.class_var.name
        #self.label_name = 'label'
#         if len(self.train_data.domain.variables)-len(self.train_data.domain.class_vars) > 1 \
#                 or len(self.train_data.domain.attributes) > 1:
#             from mlstudiosdk.exceptions.ExceptionBase import Error
#             raise Error("too many features")
        self.sentence_name = self.train_data.domain.attributes[0].name
 
    def add_commentid(self, df_input):

        df_input = df_input.reset_index(drop=True)
        return df_input
    
    def merge_possibility(self, data_len, possibility, legal_test_data_id):
        # combine neg/posi possibilty with illegal
        pos_new = [[None]*2]*data_len
        for id_index in range(len(possibility)):
            id = legal_test_data_id[id_index]
            pos_new[id] = possibility[id_index]
        return np.array(pos_new)
      
    def merge_data(self,origin, result):
        # result table to origin table's meta
        domain = origin.domain
        attributes, classes = domain.attributes, domain.class_vars
        meta_attrs = domain.metas
        new_domain = Domain(attributes, classes, meta_attrs+result.domain.variables)
        new_table = Table.from_table(new_domain, origin)
        for attr in result.domain:
            new_table.get_column_view(attr)[0][:] = result.get_column_view(attr)[0][:]
        return new_table

    #define a program for text reprocessing
    def text_split(self,data,data_type):
        #get the columns list
        colums_names=data.columns.values.tolist()
        #filter'id','content' columns to get the column names of multiple tags
        colums_names_list=colums_names[2:]
        #get the 'content' columns
        text_name=colums_names[1]
        #get 'content' column text,convert to list
        text_set=data[[text_name]].values.tolist()
        #get the stopwords list
        stopwords = {}.fromkeys([ line.rstrip() for line in codecs.open(r"stopwords.txt",'r','utf-8-sig') ])
       #split data by data_type
        if data_type =='trainingset' or data_type =='validationset': 
        #loop through the operation

            with open(r'fasttext_'+data_type+'.txt','w',encoding='utf-8')as f: 
               
                for i in range(len(text_set)):               
                    temp1=' '.join(jieba.cut(str(text_set[i])))
                    texts1 = [word for word in temp1.split() if word not in stopwords]
                    texts1 = " ".join(texts1)

                    #Add the prefix '__label__' to the label column
                    temp2='__label__'+str(list(data[self.label_name])[i])+' ,'
                    temp3=texts1+' '+temp2                 
                    f.write(temp3+'\n')            
            f.close() 
        else:   
            #deal with the data_type'testset'
            with open(r'fasttext_'+data_type+'.txt','w',encoding='utf-8')as f:   
                for i in range(len(text_set)):
                    #Note: the testset dataset has only a context column and no prefix '__label__'
                    temp1=' '.join(jieba.cut(str(text_set[i])))
                    texts1 = [word for word in temp1.split() if word not in stopwords]
                    texts1 = " ".join(texts1)
                    #texts1 = str(re.findall(u'[\u4e00-\u9fa5-\d+\.\d]+'," ".join(texts1)))                    
                    temp3=texts1
                    f.write(temp3+'\n')                   
            f.close()
        return colums_names_list
            
    def run(self):
        self.get_name()
        
        test_data = table2df(self.test_data) 
       
#         try:
#             test_data.iloc[:, :-len(self.test_data.domain.class_vars)] = \
#                 test_data.iloc[:, :-len(self.test_data.domain.class_vars)].astype('float')
#         except Exception as e:
#             print(e)
#             raise Error('data must be numeric')
        test_data = test_data.dropna(subset=[self.label_name])
        test_data, test_label_domain = self.label_str2number(test_data)
        test_data = self.add_commentid(test_data)

        if len(test_data) != len(self.test_data):
            raise Error("test data missing label")
       
        test_data_raw = test_data # test data may delete some records
        y_test_visual = test_data_raw[self.label_name]
        train_data = table2df(self.train_data) 
        train_data = self.add_commentid(train_data)
#         try:
#             train_data.iloc[:, :-len(self.train_data.domain.class_vars)] = \
#                 train_data.iloc[:, :-len(self.train_data.domain.class_vars)].astype('float')
#         except Exception as e:
#             print(e)
#             raise Error('data must be numeric')
        if len(test_data.columns) != len(train_data.columns):
            raise Error('train data and test data must match')
        train_data = train_data.dropna(subset=[self.label_name])              
        train_data, self.label_domain = self.label_str2number(train_data)
#        print("self.label_domain",self.label_domain)
        Traina,validationa = train_test_split(train_data, train_size = 0.8, random_state=1234) 
        Traina = self.add_commentid(Traina)
        validationa = self.add_commentid(validationa)
        self.text_split(Traina,'trainingset')
        legal_test_data_id= list(test_data.index)
        test_split = test_data.drop([self.label_name],axis=1)
        self.text_split(test_data,'testset')
        self.text_split(test_split,'test_split')
        self.text_split(validationa,'validationset')       
        #training the model
        self.classifier=fasttext.supervised('fasttext_trainingset.txt','fasttext_test.model',label_prefix='__label__',
                                        thread=32,epoch=50,lr=0.1,dim=200,bucket=5000000)
      
        train_result=self.classifier.test('fasttext_trainingset.txt')
        train_score=train_result.precision
        validation_result=self.classifier.test('fasttext_validationset.txt')    
        val_score=validation_result.precision
        test_set = open('fasttext_test_split.txt','r',encoding='utf-8-sig')
        predict_label =self.classifier.predict(test_set)    
        int_index = []
        for index in predict_label:
            int_index.append(int(index[0]))

         
        test_set1 = open('fasttext_test_split.txt','r',encoding='utf-8-sig')
        wind_pre_prob=self.classifier.predict_proba(test_set1)
        wind_pre_prob_index = []
        for index in wind_pre_prob:
            wind_pre_prob_index.append(index[0][1])

        
        """-----------calculate probility-------------------------------"""

        metas = [DiscreteVariable('predict', self.label_domain),
                 ContinuousVariable('negative probability'),
                 ContinuousVariable('positive probability')]

        new_two_prob_positive = []
        new_two_prob_negative = []

        for index, item in enumerate(wind_pre_prob_index):
                new_two_prob_negative.append(1 - item)
                new_two_prob_positive.append(item)

        wind_pre_prob = np.array([np.array(new_two_prob_negative), np.array(new_two_prob_positive)])
        wind_pre_prob = wind_pre_prob.T
        total_pre_prob = self.merge_possibility(len(test_data_raw), wind_pre_prob, legal_test_data_id)
        cols = [np.array(int_index).reshape(-1,1), wind_pre_prob]
        aa=np.array(int_index).reshape(-1, 1)

         
        tbl = np.column_stack((np.array(int_index).reshape(-1, 1), wind_pre_prob))

        res = Table.from_numpy(Domain(metas), tbl)
        final_result = self.merge_data(self.test_data,res)
        
        
        results = None
        N = len(test_data)  # note that self.test_data is orange Table
        results = Results(self.test_data[legal_test_data_id], store_data=True)
        results.folds = None
        results.row_indices = np.arange(N)
        results.actual = np.array(y_test_visual[legal_test_data_id])
        results.predicted = np.array([int_index])

        results.probabilities = np.array([wind_pre_prob])  
        # results.probabilities = wind_pre_prob

        if len(self.label_domain)-1 > 2:
            results.probabilities = None
        
        print("predicted", results.predicted[0])
        print("actual", results.actual)

        results.learner_names = ['Multiple_text_Classifier']
        # self.send("Predictions", predictions)
        self.send("Evaluation Results", results)

        metric_frame = MetricFrame([[val_score], [train_score]], index=["hyperparam_searcher", "modeler"],
                                   columns=[MetricType.ACCURACY.name])
        self.send("Metric Score", metric_frame)
        self.send("Columns", cols)
        self.send("Metas", metas)
        self.send("Metric", MetricType.ACCURACY)

        # print(MetricType.ACCURACY)
        print("metric_frame",metric_frame)
        print("result:",results.predicted, results.predicted.shape)        
         
        self.send('News', final_result)
        self.remove_files('.', 'fasttext_*.txt', show = True)


    def rerun(self):
        test_data = table2df(self.test_data)
        test_data = self.add_commentid(test_data)
#         test_data = test_data.dropna(subset=[self.label_name])
        text_split=test_data
        test_data_raw = test_data # test data may delete some records
#         text_split= test_data.drop([self.label_name], axis=1)
        self.text_split(text_split,'test_split')
        legal_test_data_id = list(test_data.index)
        #loading the fasttext model
        classifier = fasttext.load_model("fasttext_test.model.bin", label_prefix="__label__")
        test_set = open('fasttext_test_split.txt','r',encoding='utf-8-sig')
        predict_label =classifier.predict(test_set)    
        predict_label_df=pd.DataFrame(predict_label)
        int_index = []
        for index in predict_label:
            int_index.append(int(index[0]))
         
        test_set1 = open('fasttext_test_split.txt','r',encoding='utf-8-sig')
        wind_pre_prob=self.classifier.predict_proba(test_set1)
        wind_pre_prob_index = []
        for index in wind_pre_prob:
            wind_pre_prob_index.append(index[0][1])


        """-----------calculate probility-------------------------------"""

        metas = [DiscreteVariable('predict', self.label_domain),
                 ContinuousVariable('negative probability'),
                 ContinuousVariable('positive probability')]

        new_two_prob_positive = []
        new_two_prob_negative = []
          
        for index, item in enumerate(wind_pre_prob_index):
                new_two_prob_negative.append(1 - item)
                new_two_prob_positive.append(item)

        wind_pre_prob = np.array([np.array(new_two_prob_negative), np.array(new_two_prob_positive)])
        wind_pre_prob = wind_pre_prob.T
        total_pre_prob = self.merge_possibility(len(test_data_raw), wind_pre_prob, legal_test_data_id)
        cols = [np.array(int_index).reshape(-1,1), wind_pre_prob]
        aa=np.array(int_index).reshape(-1, 1)

         
        tbl = np.column_stack((np.array(int_index).reshape(-1, 1), wind_pre_prob))

        res = Table.from_numpy(Domain(metas), tbl)
        final_result = self.merge_data(self.test_data,res)
        
        self.send("News", final_result)
        self.send("Metric Score", None)
        self.send("Metas", metas)
        self.send("Columns", cols)
        print('rerun')
        self.remove_files('.', 'fasttext_*.txt', show = True)
                          


