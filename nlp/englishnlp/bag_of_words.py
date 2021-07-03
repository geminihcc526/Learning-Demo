import pandas as pd
train = pd.read_csv('alldata/labeledTrainData.tsv', header=0, delimiter='\t', quoting=3)
print(train.shape)
print(train.columns.values)
print(train.head())
print(train['review'][0])

test = pd.read_csv('alldata/testData.tsv', header=0, delimiter='\t', quoting=3)
print(test.shape)
print(test.head())

train_split = train['review'][0].split(",")
for str in train_split:
    print(str)

# data cleaning
from bs4 import BeautifulSoup
# 在一条评论上初始化一个BeautifulSoup对象
# 用beautifulsoup来清洗html标签
example1 = BeautifulSoup(train['review'][0], 'lxml')

# 比较一下原始的文本和处理过后的文本的差别，通过调用get_text()得到处理后的结果
print(train['review'][0])
print()
print(example1.get_text())

import re
letters_only = re.sub('[^a-zA-Z]', # The pattern to search for
                      ' ',         # The pattern to repalce it with
                      example1.get_text()) # The text to search
print(letters_only)

#大写变小写
lower_case = letters_only.lower() # Convert to lower case

#tokenization
words = lower_case.split() # Split into words

#stop words removal
from nltk.corpus import stopwords # import the stop word list
print(stopwords.words('english')[:10])

#从评论取出stop words
words = [w for w in words if not w in stopwords.words('english')]
print(words[:10])

