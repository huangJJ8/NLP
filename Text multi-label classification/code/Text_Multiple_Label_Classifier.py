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
import jieba
import os
import glob #find file directories and files
import fasttext  #model
import codecs
import re
import warnings
warnings.filterwarnings('ignore')


class Multiple_text_Classifier(LComponent):
    name = "Text_Classifier"
    
    inputs = [("Train Data", mlstudiosdk.modules.algo.data.Table, "set_traindata"),
              ("Test Data", mlstudiosdk.modules.algo.data.Table, "set_testdata")]
    outputs = [("Metric Score", MetricFrame),
               ("Data_OUT", mlstudiosdk.modules.algo.data.Table),
               ("Evaluation Results",mlstudiosdk.modules.algo.data.Table)
              ]     
    
    def __init__(self):
        super().__init__()
        self.train_data = None
        self.test_data = None

    
    def set_traindata(self, data):
        self.train_data = data

    def set_testdata(self, data):
        self.test_data = data

#     def get_name(self):
#         # get label & sentence columns name
#         self.label_name = self.train_data.domain.class_var.name
#         self.sentence_name = self.train_data.domain.attributes[0].name

#     def label_str2number(self,data):
#         label_str_raw = list(set(data[self.label_name]))
#         label_str = label_str_raw+['illegal data']
#         format = lambda x: label_str .index(x)
#         data[self.label_name] = data[self.label_name].map(format)
#         return data,label_str_raw+['illegal data']

#     def label_number2str(self,data,label_str):
#         format = lambda x: label_str[x]
#         data[self.label_name] = data[self.label_name].map(format)
#         return data


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
    
    #define a program for text reprocessing
    def text_split(self,data,data_type):
        #get the columns list
        colums_names=data.columns.values.tolist()
        #filter'id','content' columns to get the column names of multiple tags
        colums_names_list=colums_names[2:]
        #获取'content' columns
        #text_name=colums_names[1]
        #get 'content' column text,convert to list
        text_set=data[['content']].values.tolist()
        #get the stopwords list
        stopwords = {}.fromkeys([ line.rstrip() for line in codecs.open(r"stopwords.txt",'r','utf-8-sig') ])
       #split data by data_type
        if data_type =='trainingset' or data_type =='validationset': 
        #loop through the operation
          for colums_name in colums_names_list:  
            ct=[]
            with open(r'fasttext_'+data_type+'_'+colums_name+'.txt','w',encoding='utf-8')as f: 
               
                for i in range(len(text_set)):               
                    temp1=' '.join(jieba.cut(str(text_set[i])))
                    texts1 = [word for word in temp1.split() if word not in stopwords]
                    texts1 = " ".join(texts1)
                    #texts1 = str(re.findall(u'[\u4e00-\u9fa5-\d+\.\d]+'," ".join(texts1)))
                    #Add the prefix '__label__' to the label column
                    temp2='__label__'+str(list(data[colums_name])[i])+' ,'
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
    #define f1_score process
    def f1_score_imitate(self,real_label, predict_label,label_class): 
        f1_score = 0
        #class_list = label_class
        #get the label classification unique values
        qty=len(list(set(label_class)))
        for num in label_class:
            tp = 0
            fp = 0
            tf = 0
            fn = 0
        for label_1 in range(len(real_label)):
            """
                                    (Predictive Label)
                                Positive           Negative
                       True        TP                TN
          (Actual Label)       
                      False        FP                FN
           
           """
            if real_label[label_1] == num and predict_label[label_1] == num:
                tp += 1
            elif real_label[label_1] == num and predict_label[label_1] != num:
                fn += 1
            elif real_label[label_1] != num and predict_label[label_1] == num:
                fp += 1
            elif real_label[label_1] != num and predict_label[label_1] != num:
                tf += 1
        if tp == 0:
            f1_score += 0
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1_score += 2 * (precision * recall) / (precision + recall)
        f1_score=f1_score /qty 
        return  f1_score     
    #define the fasttext model
    def make_model_structure(self,trainingset, validationset,thread=32,epoch=50,lr=0.1,dim=200,bucket=5000000 ):
        classifier=fasttext.supervised(trainingset,'train.model',label_prefix='__label__',thread=32,epoch=50,lr=0.1,dim=200,bucket=5000000)
        validation_set = open(validationset,'r',encoding='utf-8-sig') 
     
        #model.summary()
        return classifier, validation_set

            
    def run(self):
#         self.get_name()
        train_data = table2df(self.train_data) 
        test_data1 = table2df(self.test_data) 
#         train_data, self.label_domain = self.label_str2number(train_data)
#         test_data, test_label_domain = self.label_str2number(test_data)
        Traina,validationa = train_test_split(train_data, train_size = 0.8, random_state=1234)    
        self.colums_names_list = self.text_split(Traina,'trainingset')
        self.text_split(validationa,'validationset')
        #original data preparation for the obfuscation matrix
        pre_columns=validationa.columns.values.tolist()[:2]
        #['\ufeffid','content']
        validationa_split=validationa[pre_columns].reset_index(drop=True)
        self.text_split(validationa_split,'validationsplit')
        self.text_split(test_data1,'testset')
        data_frame_nll = pd.DataFrame()
        for item in self.colums_names_list:
            
            #Training the fasttext model
            classifier,validation_set= self.make_model_structure('fasttext_trainingset_'+item+'.txt',
                                                                'fasttext_validationset_'+item+'.txt' )
            #Once the model was trained, we can evaluate it by computing the precision at 1 (P@1) and the recall on a test set using classifier.test function            
            result=classifier.test('fasttext_validationset_'+item+'.txt')
            #In order to obtain the most likely label for a list of text, we can use classifer.predict method
            predict_label = pd.DataFrame(classifier.predict(validation_set),columns=[item])
            #Construct obfuscation matrix real\predict_label
            validationa_split['predict_label']=predict_label
            validationa_split['real_label']=validationa[item].tolist()
            #To figure out F1 score,convert real_label,predict_label to list
            real_label=validationa[item].astype(np.float64).tolist()
            #get the label classification unique values
            label_class=list(set(real_label))
            num=len(label_class)
            predict_label=np.array(predict_label[item]).astype(np.float64)
            predict_label=predict_label.tolist()
            #figure out Fl_score
            f1_score=self.f1_score_imitate(real_label, predict_label,label_class)
            Score=MetricFrame([[result.precision], [result.recall],[result.recall]], index=["Precision", "Recall","f1_score"],
                                   columns=[MetricType.ACCURACY.name])
            Score.to_csv('fasttext_class_'+item+'_score.csv',index=True)  
            self.send("Metric Score", Score)            
            #Output the predict_proba result of validation set
            validationa_split_t = open('fasttext_validationsplit.txt','r',encoding='utf-8-sig') 
            result_t1= pd.DataFrame(classifier.predict_proba(validationa_split_t,k=num))
            validationa_split['label_name']=item
              
            
            #split the predict_proba result to fit the Table function input format
            list_2=[]
            for name in result_t1.columns.tolist():
                data_frame = pd.DataFrame()
                for subclass in label_class: 
                    list1 = []
                    for every_name in result_t1[name]:
                        if every_name[0] == str(subclass):
                            list1.append(every_name[1])
                        else:
                            list1.append(0)
                    data_frame[subclass] = list1
                list_2.append(data_frame)
            for i in range(1,len(label_class)): 
                list_2[0] += list_2[i]
            predict_proba_detail=list_2[0] 
            
            #Summary the validationset output for the obfuscation matrix
            label_sum=pd.concat([validationa_split,predict_proba_detail],axis=1) 
            
            #Output the classification result collection of the testset
            testset_sum = open('fasttext_testset.txt','r',encoding='utf-8-sig') 
            result_t2= pd.DataFrame(classifier.predict(testset_sum),columns=[item])
            test_data1[item]=result_t2
            testset_sum.close()    
            data_frame_nll=pd.concat([data_frame_nll,label_sum],axis=0) 
            validationa_split_t.close()
        data_frame_nll.to_csv('fasttext_class_validationset_pred.csv',index=False)  
        validation_pro_label=df2table(data_frame_nll)
        self.send('Evaluation Results', validation_pro_label)
        test_data1.to_csv('fasttext_testset_predict_result.csv',index=False)  
        output=df2table(test_data1)               
        self.send('Data_OUT', output)        

    def rerun(self):
        test_data2 = table2df(self.test_data) 
        self.text_split(test_data2,'testset')
        for item in self.colums_names_list:
            #Training the fasttext model
            classifier,validation_set = self.make_model_structure('fasttext_trainingset_'+item+'.txt', 'fasttext_validationset_'+item+'.txt' )
            #Output the classification result collection of the testset
            testset3 = open('fasttext_testset.txt','r',encoding='utf-8-sig')
            result_t3= pd.DataFrame(classifier.predict(testset3),columns=[item])
            test_data2[item]=result_t3
            testset3.close()
        test_data2.to_csv('fasttext_testset_rerun_result.csv',index=False)
        test_data4=df2table(test_data2)
        output=test_data4   
        self.send('Data_OUT', output)
        
        
if __name__ == '__main__':
    s = Multiple_text_Classifier()
    s.train_data = Table("/mnt/data/input/AI_Template/Sentiment/fasttext_training1113.csv")
    #s.train_data = Table("/mnt/data/input/AI_Template/Sentiment/training_1103.csv")
    s.test_data = Table("/mnt/data/input/AI_Template/Sentiment/fasttext_test1113.csv")
    #s.test_data = Table("/mnt/data/input/AI_Template/Sentiment/test_1103.csv")
    s.run()
    s.rerun()   
    s.remove_files('.', 'fasttext_*.txt', show = True) 