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
import jieba   #add jieba by chenjing
import jieba.posseg as pseg
import codecs
import os

import mlstudiosdk.modules.components.nlp.Text_Extract_Keywords_Util as util
warnings.filterwarnings('ignore')


def get_mlstudiosdk_path():
    return os.path.dirname(mlstudiosdk.__file__)


def get_dataset_path(name):
    dataset_path = os.path.join(get_mlstudiosdk_path(), 'dataset', 'dataset_for_nlp',"Keyword_Extraction",
                                name)
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

    def segment_sentences(self, sentences, lower=True, use_stop_words=True, use_speech_tags_filter=False):
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
        res = [util.as_text(text)]

        util.debug(res)
        util.debug(self.delimiters)

        for sep in self.delimiters:
            text, res = res, []
            for seq in text:
                res += seq.split(sep)
        res = [s.strip() for s in res if len(s.strip()) > 0]
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


class TF_IDF():
    def __init__(self):
        self.idf_file = get_dataset_path('IDF.txt')
        self.idf_dict, self.common_idf = self.load_idf()

    def build_wordsdict(self, text):
        word_dict = {}
        candi_words = []
        candi_dict = {}
        for word in pseg.cut(text):
            if word.flag[0] in ['an', 'i', 'j', 'l', 'n', 'nr', 'nrfg', 'ns', 'nt', 'nz', 't', 'v', 'vd', 'vn',
                                'eng'] and len(word.word) > 1:  # ['n', 'v', 'a']
                candi_words.append(word.word)
            if word.word not in word_dict:
                word_dict[word.word] = 1
            else:
                word_dict[word.word] += 1
        #         print('word_dict:',word_dict)
        count_total = sum(word_dict.values())
        for word, word_count in word_dict.items():
            if word in candi_words:
                candi_dict[word] = word_count / count_total
            else:
                continue

        return word_dict, candi_dict
    """ add the build_wordsdict1 to figure out words frequence by chenjing """
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
        word_dict, candi_dict = self.build_wordsdict(text)
        for word, word_tf in candi_dict.items():
            word_idf = self.idf_dict.get(word, self.common_idf)
            word_tfidf = word_idf * word_tf
            keywords_dict[word] = word_tfidf
        keywords_dict = sorted(keywords_dict.items(), key=lambda asd: asd[1], reverse=True)

        return keywords_dict[:num_keywords]

    def load_idf(self):
        idf_dict = {}
        for line in open(self.idf_file, encoding='utf-8'):
            word, freq = line.strip().split(' ')
            idf_dict[word] = float(freq)
        common_idf = sum(idf_dict.values()) / len(idf_dict)

        return idf_dict, common_idf


class TextRank4Keyword(object):

    def __init__(self, stop_words_file=None,
                 allow_speech_tags=util.allow_speech_tags,
                 delimiters=util.sentence_delimiters):
        """
        Keyword arguments:
        stop_words_file  --  str，指定停止词文件路径（一行一个停止词），若为其他类型，则使用默认停止词文件
        delimiters       --  默认值是`?!;？！。；…\n`，用来将文本拆分为句子。

        Object Var:
        self.words_no_filter      --  对sentences中每个句子分词而得到的两级列表。
        self.words_no_stop_words  --  去掉words_no_filter中的停止词而得到的两级列表。
        self.words_all_filters    --  保留words_no_stop_words中指定词性的单词而得到的两级列表。
        """
        self.text = ''
        self.keywords = None

        self.seg = Segmentation(stop_words_file=stop_words_file,
                                allow_speech_tags=allow_speech_tags,
                                delimiters=delimiters)

        self.sentences = None
        self.words_no_filter = None  # 2维列表
        self.words_no_stop_words = None
        self.words_all_filters = None

    def analyze(self, text,
                window=2,
                lower=False,
                vertex_source='all_filters',
                edge_source='no_stop_words',
                pagerank_config={'alpha': 0.85, }):
        """分析文本

        Keyword arguments:
        text       --  文本内容，字符串。
        window     --  窗口大小，int，用来构造单词之间的边。默认值为2。
        lower      --  是否将文本转换为小写。默认为False。
        vertex_source   --  选择使用words_no_filter, words_no_stop_words, words_all_filters中的哪一个来构造pagerank对应的图中的节点。
                            默认值为`'all_filters'`，可选值为`'no_filter', 'no_stop_words', 'all_filters'`。关键词也来自`vertex_source`。
        edge_source     --  选择使用words_no_filter, words_no_stop_words, words_all_filters中的哪一个来构造pagerank对应的图中的节点之间的边。
                            默认值为`'no_stop_words'`，可选值为`'no_filter', 'no_stop_words', 'all_filters'`。边的构造要结合`window`参数。
        """

        # self.text = util.as_text(text)
        self.text = text
        self.word_index = {}
        self.index_word = {}
        self.keywords = []
        self.graph = None

        result = self.seg.segment(text=text, lower=lower)
        self.sentences = result.sentences
        self.words_no_filter = result.words_no_filter
        self.words_no_stop_words = result.words_no_stop_words
        self.words_all_filters = result.words_all_filters

        util.debug(20 * '*')
        util.debug('self.sentences in TextRank4Keyword:\n', ' || '.join(self.sentences))
        util.debug('self.words_no_filter in TextRank4Keyword:\n', self.words_no_filter)
        util.debug('self.words_no_stop_words in TextRank4Keyword:\n', self.words_no_stop_words)
        util.debug('self.words_all_filters in TextRank4Keyword:\n', self.words_all_filters)

        options = ['no_filter', 'no_stop_words', 'all_filters']

        if vertex_source in options:
            _vertex_source = result['words_' + vertex_source]
        else:
            _vertex_source = result['words_all_filters']

        if edge_source in options:
            _edge_source = result['words_' + edge_source]
        else:
            _edge_source = result['words_no_stop_words']

        self.keywords = util.sort_words(_vertex_source, _edge_source, window=window, pagerank_config=pagerank_config)

    def get_keywords(self, num=6, word_min_len=1):
        """获取最重要的num个长度大于等于word_min_len的关键词。

        Return:
        关键词列表。
        """
        result = []
        count = 0
        for item in self.keywords:
            if count >= num:
                break
            if len(item.word) >= word_min_len:
                result.append(item)
                count += 1
        return result

    def get_keyphrases(self, keywords_num=12, min_occur_num=2):
        """获取关键短语。
        获取 keywords_num 个关键词构造的可能出现的短语，要求这个短语在原文本中至少出现的次数为min_occur_num。

        Return:
        关键短语的列表。
        """
        keywords_set = set([item.word for item in self.get_keywords(num=keywords_num, word_min_len=1)])
        keyphrases = set()
        for sentence in self.words_no_filter:
            one = []
            for word in sentence:
                if word in keywords_set:
                    one.append(word)
                else:
                    if len(one) > 1:
                        keyphrases.add(''.join(one))
                    if len(one) == 0:
                        continue
                    else:
                        one = []
            # 兜底
            if len(one) > 1:
                keyphrases.add(''.join(one))

        return [phrase for phrase in keyphrases
                if self.text.count(phrase) >= min_occur_num]


"""----------Main Function--------"""


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
        ("Jsondata", dict)
    ]

    """ changed key_dict moved to here and changed name """
    # n_keywords = Setting(5, {"type": "number", "enum": [1, 2, 3, 4, 5, 6], "minimum": 0, "exclusiveMinimum": True})
    n_keywords = Setting(5, {"type": "integer", "minimum": 0, "exclusiveMinimum": True})

    def __init__(self):
        super().__init__()
        self.train_data = None
        self.keywords_num = 5

    def set_traindata(self, data):
        self.train_data = data

    def get_keywords_num(self):
        # get keywords_num
        # key_dict = Setting(5, {"type": "number", "enum": [1, 2, 3, 4, 5, 6], "minimum": 0, "exclusiveMinimum": True})
        self.keywords_num = self.n_keywords

    def run(self):
        # text = codecs.open('test.csv', 'r', 'utf-8').read()
        self.get_keywords_num()
        train_data = table2df(self.train_data)
        
        if train_data.index.size == 0:
            raise AttributesError_new("Missing data, the input dataset should have two rows and one column "
                              "(the first row is column name,the second row is the single text.)")
        elif train_data.columns.size > 1:
            raise AttributesError_new("Input data should have only one column")
        elif train_data.index.size > 1:
            raise AttributesError_new("The single text should be filled into one row")
        """ use 1 col and 1 row data """    
        train_data = str(train_data.values[0][0])
        # use Textrank to get the keywords
        tr4w = TextRank4Keyword()
        tr4w.analyze(text=train_data, lower=True, window=5)
        list1 = []
        list2 = []
        list3 = []
        list4 = []
        list5 = []
        list6 = []
        result = []
        for item in tr4w.get_keywords(self.keywords_num * 2, word_min_len=2):
            list1.append([item.word, item.weight])
            list6.append(item.word)
        self.keywords_set = list(set(list6))
        #         print('keywords_set:',self.keywords_set)
        #         print(list1)

        # use TF_IDF to get the keywords
        tfidfer = TF_IDF()
        word_dict, candi_dict = tfidfer.build_wordsdict(train_data)
        """ add the build_wordsdict1 to figure out words frequence by chenjing """
        word_dict1=tfidfer.build_wordsdict1(train_data)
        for keyword in tfidfer.extract_keywords(train_data, self.keywords_num * 2):
            list2.append(list(keyword))
        #         print(list2)
        for i in range(len(list1)):
            for j in range(len(list2)):
                if list1[i][0] == list2[j][0]:
                    list3.append(list1[i])
        for i in list1:
            if i not in list3:
                list4.append(i)

        length = len(list3)
        if length >= self.keywords_num:
            for i in range(self.keywords_num):
                result.append(list3[i])
        else:
            result = list3
            if len(list4) >= self.keywords_num - length:
                for i in range(self.keywords_num - length):
                    list3.append(list4[i])
                    result = list3
        result.sort(key=lambda k: k[1], reverse=True)
        for i in range(len(result)):
            for word, word_tf in word_dict1.items():
                if result[i][0] == word:
                    result[i].append(int(word_tf))
        # print('result1：',result)

        for i in range(len(result)):
            list7 = []
            list7.append(result[i][0])
            #             print('list7:',list7)
            mapping = list(map(lambda x: self.keywords_set.index(x), list7))
            #             print('mapping:',mapping)
            result[i][0] = mapping[0]
        # print('result2：',result)

        metas = [DiscreteVariable('keywords', self.keywords_set),
                 ContinuousVariable('weight'),
                 ContinuousVariable('word_frequency')]
        #         print('Domain(metas):',Domain(metas))
        #         listma=[[1,2,3],[4,5,6],[3,5,6]]
        #         print(listma)
        domain = Domain(metas)
        #         print('domain.attributes:',domain.attributes)
        #         print('domain.class_vars:',domain.class_vars)
        final_result = Table.from_list(domain, result)
        #         final_result=Table.from_list(Domain(metas),listma)
        # print('final_result:',final_result)
        json_res = {}
        temp_lst = []
        fields = ['name', 'weight', 'count']
        for i in result:
            temp_dir = {}
            for j, k in enumerate(i):
                if j != 0:
                    temp_dir[fields[j]] = k
                else:
                    temp_dir[fields[j]] = self.keywords_set[k]
            temp_lst.insert(0, temp_dir)
        json_res['visualization_type'] = "keywords"
        json_res['results'] = temp_lst

        json_res["chartXName"] = 'weight'
        json_res["chartYName"] = 'name'
        json_res["tableCols"] = ['name', 'count']

        # print(json_res)
        self.send('Jsondata', json_res)
        # json_dicts = json.dumps(json_res, indent=4)
        # with open('c.json', 'w') as f:
        #    json.dump(json_res, f, ensure_ascii=False, indent=4)
        # f.close()
        # print("加载入文件完成...")
        # print(json_dicts)
        self.send('News', final_result)
        self.send("Metas", metas)
