输入训练/测试数据字段格式

id:


type: str
每个句子的唯一标识符，不可重复


content：


type: str
句子的内容.
样例:  "第一次吃油泼面，哪里的风味呢！原本以为油泼面是很辣很辣的，这就很过瘾，你本人是比较能吃辣的人，早上去吃，竟然不是所想那样，油泼面并不辣，而是特别的香。"


label:


type: str/int
每个句子对应的label，目前只支持编码后的label. 暂不支持文本类型的label.可以设置多列label,比如说：“location_traffic_convenience”，“location_easy_to_find”等。
label的数值样例：“0，1，2，3，4”。
注意：如果设置多个label，保证多列label的数值分类一致，例如均分为：0,1,2,3 四类。
若label格式为 ”财经， 综艺， 军事“ 等文本类型，需预先将其转化为数字编码.