#-*- encoding:utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import sys
try:
    sys.reload(sys)
    sys.setdefaultencoding('utf-8')
except:
    pass

import mlstudiosdk.modules
import mlstudiosdk.solution_gallery
from mlstudiosdk.modules.components.component import LComponent
from mlstudiosdk.modules.components.settings import Setting
from mlstudiosdk.modules.components.utils.orange_table_2_data_frame import table2df
from mlstudiosdk.modules.algo.data import Domain, Table
from mlstudiosdk.modules.algo.evaluation import Results
from mlstudiosdk.modules.algo.data.variable import DiscreteVariable, ContinuousVariable
from mlstudiosdk.modules.utils.itemlist import MetricFrame
import warnings

import jieba.posseg as pseg
import codecs
import os
import re

import util  #modify the code for test
#import mlstudiosdk.modules.components.nlp.Text_Extract_Keywords_Util as util
warnings.filterwarnings('ignore')


def get_mlstudiosdk_path():
    return os.path.dirname(mlstudiosdk.__file__)


def get_dataset_path(name):
    #modify the code for test
    dataset_path =os.path.join('/mnt/data/input/TextRank_Summary',name) 
    #os.path.join(get_mlstudiosdk_path(), 'dataset', 'dataset_for_nlp',"Keyword_Extraction",name)
    return dataset_path


def get_default_stop_words_file():
    return get_dataset_path("stopwords.txt")


class WordSegmentation(object):
    """ 分词 """

    def __init__(self, stop_words_file=None, allow_speech_tags=util.allow_speech_tags):
        """
        Keyword arguments:
        stop_words_file    -- 保存停止词的文件路径，utf8编码，每行一个停止词。若不是str类型，则使用默认的停止词
        allow_speech_tags  -- 词性列表，用于过滤
        """

        allow_speech_tags = [util.as_text(item) for item in allow_speech_tags]

        self.default_speech_tag_filter = allow_speech_tags
        self.stop_words = set()
        self.stop_words_file = get_default_stop_words_file()
        if type(stop_words_file) is str:
            self.stop_words_file = stop_words_file
        for word in codecs.open(self.stop_words_file, 'r', 'utf-8', 'ignore'):
            self.stop_words.add(word.strip())

    def segment(self, text, lower=True, use_stop_words=True, use_speech_tags_filter=False):
        """对一段文本进行分词，返回list类型的分词结果

        Keyword arguments:
        lower                  -- 是否将单词小写（针对英文）
        use_stop_words         -- 若为True，则利用停止词集合来过滤（去掉停止词）
        use_speech_tags_filter -- 是否基于词性进行过滤。若为True，则使用self.default_speech_tag_filter过滤。否则，不过滤。
        """
        text = util.as_text(text)
        jieba_result = pseg.cut(text)

        if use_speech_tags_filter == True:
            jieba_result = [w for w in jieba_result if w.flag in self.default_speech_tag_filter]
        else:
            jieba_result = [w for w in jieba_result]

        # 去除特殊符号
        word_list = [w.word.strip() for w in jieba_result if w.flag != 'x']
        word_list = [word for word in word_list if len(word) > 0]

        if lower:
            word_list = [word.lower() for word in word_list]

        if use_stop_words:
            word_list = [word.strip() for word in word_list if word.strip() not in self.stop_words]

        return word_list

    def segment_sentences(self, sentences, lower=True, use_stop_words=False, use_speech_tags_filter=False):
        """将列表sequences中的每个元素/句子转换为由单词构成的列表。

        sequences -- 列表，每个元素是一个句子（字符串类型）
        """

        res = []
        for sentence in sentences:
            res.append(self.segment(text=sentence,
                                    lower=lower,
                                    use_stop_words=use_stop_words,
                                    use_speech_tags_filter=use_speech_tags_filter))
        return res


class SentenceSegmentation(object):
    """ 分句 """

    def __init__(self, delimiters=util.sentence_delimiters):
        """
        Keyword arguments:
        delimiters -- 可迭代对象，用来拆分句子
        """
        self.delimiters = set([util.as_text(item) for item in delimiters])
    
    def segment(self, text):
#         print('str(text)',str(text))
        res = re.split(r"([?!;？！。；...\n.?!\"])", str(text))
#         print('ress',res)
        res = ["".join(i) for i in zip(res[0::2],res[1::2])]
        res = [s for s in res if len(s.strip()) > 1] 
#         print('res1',res)
        return res     

class Segmentation(object):

    def __init__(self, stop_words_file=None,
                 allow_speech_tags=util.allow_speech_tags,
                 delimiters=util.sentence_delimiters):
        """
        Keyword arguments:
        stop_words_file -- 停止词文件
        delimiters      -- 用来拆分句子的符号集合
        """
        self.ws = WordSegmentation(stop_words_file=stop_words_file, allow_speech_tags=allow_speech_tags)
        self.ss = SentenceSegmentation(delimiters=delimiters)

    def segment(self, text, lower=False):
        text = util.as_text(text)
        sentences = self.ss.segment(text)
        words_no_filter = self.ws.segment_sentences(sentences=sentences,
                                                    lower=lower,
                                                    use_stop_words=False,
                                                    use_speech_tags_filter=False)
        words_no_stop_words = self.ws.segment_sentences(sentences=sentences,
                                                        lower=lower,
                                                        use_stop_words=True,
                                                        use_speech_tags_filter=False)

        words_all_filters = self.ws.segment_sentences(sentences=sentences,
                                                      lower=lower,
                                                      use_stop_words=True,
                                                      use_speech_tags_filter=True)

        return util.AttrDict(
            sentences=sentences,
            words_no_filter=words_no_filter,
            words_no_stop_words=words_no_stop_words,
            words_all_filters=words_all_filters
        )

class TextRank4Sentence(object):
    
    def __init__(self, stop_words_file = None, 
                 allow_speech_tags = util.allow_speech_tags,
                 delimiters = util.sentence_delimiters):
        """
        Keyword arguments:
        stop_words_file  --  str，停止词文件路径，若不是str则是使用默认停止词文件
        delimiters       --  默认值是`?!;？！。；…\n`，用来将文本拆分为句子。
        
        Object Var:
        self.sentences               --  由句子组成的列表。
        self.words_no_filter         --  对sentences中每个句子分词而得到的两级列表。
        self.words_no_stop_words     --  去掉words_no_filter中的停止词而得到的两级列表。
        self.words_all_filters       --  保留words_no_stop_words中指定词性的单词而得到的两级列表。
        """
        self.seg = Segmentation(stop_words_file=stop_words_file,
                                allow_speech_tags=allow_speech_tags,
                                delimiters=delimiters)
        
        self.sentences = None
        self.words_no_filter = None     # 2维列表
        self.words_no_stop_words = None
        self.words_all_filters = None
        
        self.key_sentences = None
        
    def analyze(self, text, lower = False, 
              source = 'no_stop_words', 
              sim_func = util.get_similarity,
              pagerank_config = {'alpha': 0.85,}):
        """
        Keyword arguments:
        text                 --  文本内容，字符串。
        lower                --  是否将文本转换为小写。默认为False。
        source               --  选择使用words_no_filter, words_no_stop_words, words_all_filters中的哪一个来生成句子之间的相似度。
                                 默认值为`'all_filters'`，可选值为`'no_filter', 'no_stop_words', 'all_filters'`。
        sim_func             --  指定计算句子相似度的函数。
        """
        
        self.key_sentences = []
        
        result = self.seg.segment(text=text, lower=lower)
        self.sentences = result.sentences
        self.words_no_filter = result.words_no_filter
        self.words_no_stop_words = result.words_no_stop_words
        self.words_all_filters   = result.words_all_filters

        options = ['no_filter', 'no_stop_words', 'all_filters']
        if source in options:
            _source = result['words_'+source]
        else:
            _source = result['words_no_stop_words']

        self.key_sentences = util.sort_sentences(sentences = self.sentences,
                                                 words     = _source,
                                                 sim_func  = sim_func,
                                                 pagerank_config = pagerank_config)

            
    def get_key_sentences(self, num = 6, sentence_min_len = 6):
        """获取最重要的num个长度大于等于sentence_min_len的句子用来生成摘要。

        Return:
        多个句子组成的列表。
        """
        result = []
        count = 0
        for item in self.key_sentences:
            if count >= num:
                break
            if len(item['sentence']) >= sentence_min_len:
                result.append(item)
                count += 1
        return result
    

"""----------Main Function--------"""
class Text_Extract_Summary(LComponent):
    category = 'Nature Language Processing'
    name = "Text Extract Summary"
    title = "Text Extract Summary"

    inputs = [("Train Data", mlstudiosdk.modules.algo.data.Table, "set_traindata")
              ]

    outputs = [
        ("News", mlstudiosdk.modules.algo.data.Table),
        ("Predictions", Table),
        ("Evaluation Results", mlstudiosdk.modules.algo.evaluation.Results),
        ("Columns", list),
        ("Metas", list),
        ("Metric Score", MetricFrame),
        ("Jsondata", dict)
    ]


    def __init__(self):
        super().__init__()
        self.train_data = None

    def set_traindata(self, data):
        self.train_data = data

    def run(self):
        train_data = table2df(self.train_data)
        if train_data.index.size == 0:
            raise AttributesError_new("Missing data, the input dataset should have two rows and one column "
                              "(the first row is column name,  the second row is the single text.)")
        elif train_data.columns.size > 1:
            raise AttributesError_new("Input data should have only one column")
        elif train_data.index.size > 1:
            raise AttributesError_new("The single text should be filled into one row")
        """ use 1 col and 1 row data """
        train_data = str(train_data.values[0][0])
        # use Textrank to get the keywords
        tr4s = TextRank4Sentence()
        tr4s.analyze(text=train_data, lower=True, source = 'all_filters')
        self.proportion_set=['20%','30%','40%','50%']
        list1=[]
        list2=[]
        list3=[]
        list4=[]
        list5=[]
        list6=[]
        result=[]
        for item in tr4s.get_key_sentences(num=4):
        #print(item.index, item.weight, item.sentence)
            list1.append([item.index, item.weight, item.sentence])
        # print('list1',list1)
        list6.append(['20%',list1[0][2]])
        list5.append(list1[0][2])

        for i in range(2):
            list2.append(list1[i])
        list2.sort(key= lambda k:k[0],reverse=False)
        list6.append(['30%',list2[0][2]+list2[1][2]])
        list5.append(list2[0][2]+list2[1][2])

        for i in range(3):
            list3.append(list1[i])
        list3.sort(key= lambda k:k[0],reverse=False)
        list6.append(['40%',list3[0][2]+list3[1][2]+list3[2][2]])
        list5.append(list3[0][2]+list3[1][2]+list3[2][2])

        for i in range(4):
            list4.append(list1[i])
        list4.sort(key= lambda k:k[0],reverse=False)
        list6.append(['50%',list4[0][2]+list4[1][2]+list4[2][2]+list4[3][2]])
        list5.append(list4[0][2]+list4[1][2]+list4[2][2]+list4[3][2])
        self.abstract_set=list5
#         print('list5',list5)   
#         print('list6',list6)   
        for i in range(len(list6)):
            list7=[]
            list8=[]
            list7.append(list6[i][0])
            list8.append(list6[i][1])
           
            mapping1=list(map(lambda x: self.proportion_set.index(x),list7))
            mapping2=list(map(lambda x: self.abstract_set.index(x),list8))
#             print('mapping1',mapping1[0])
#             print('mapping2',mapping2[0])
            result.append([mapping1[0],mapping2[0]])
#         print('list6:',list6)
#         print('result:',result)     
        
        metas = [DiscreteVariable('proportion',self.proportion_set),
                 DiscreteVariable('abstract',self.abstract_set)]

        domain = Domain(metas)
        #         print('domain.attributes:',domain.attributes)
        #         print('domain.class_vars:',domain.class_vars)
        final_result = Table.from_list(domain, result)
        self.send('News', final_result)
        self.send("Metas", metas)
     
#         print('result:',result)
#         print('final_result:',final_result)
        json_res = {}
        temp_lst = []
        fields = ['proportion', 'abstract']
        for i in result:
            temp_dir = {}
            for j, k in enumerate(i):
                if j == 0:
                    temp_dir[fields[j]] = self.proportion_set[k]
                else:
                    temp_dir[fields[j]] = self.abstract_set[k]
            temp_lst.insert(0, temp_dir)
        json_res['visualization_type'] = "summary"
        json_res['results'] = temp_lst

        json_res["chartXName"] = 'proportion'
        json_res["chartYName"] = 'abstract'
#         json_res["tableCols"] = ['name', 'count']

#         print('json_res:',json_res)
        self.send('Jsondata', json_res)
