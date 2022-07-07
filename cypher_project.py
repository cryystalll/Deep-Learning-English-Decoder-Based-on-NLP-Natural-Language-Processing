#!/usr/bin/env python
# coding: utf-8

# In[1]:


englishLetterFreq = {'E': 12.70, 'T': 9.06, 'A': 8.17, 'O': 7.51, 'I': 6.97, 'N': 6.75, 'S': 6.33, 'H': 6.09, 'R': 5.99, 'D': 4.25, 'L': 4.03, 'C': 2.78, 'U': 2.76, 'M': 2.41, 'W': 2.36, 'F': 2.23, 'G': 2.02, 'Y': 1.97, 'P': 1.93, 'B': 1.29, 'V': 0.98, 'K': 0.77, 'J': 0.15, 'X': 0.15, 'Q': 0.10, 'Z': 0.07}
sortlet = 'ETAOINSHRDLCUMWFGYPBVKJXQZ'
let = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def getcount(message):
    
        letterCount = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'H': 0, 'I': 0, 'J': 0, 'K': 0, 'L': 0, 'M': 0, 'N': 0, 'O': 0, 'P': 0, 'Q': 0, 'R': 0, 'S': 0, 'T': 0, 'U': 0, 'V': 0, 'W': 0, 'X': 0, 'Y': 0, 'Z': 0}

        for letter in message.upper():
            if letter in let:
                letterCount[letter] += 1

        return letterCount

    
def getkey(x):
        return x[0]


# In[2]:


def getorder(message):

    ltf = getcount(message)
    ftl = {}

    for letter in let:
            if ltf[letter] not in ftl:
                    ftl[ltf[letter]] = [letter]
            else:
                     ftl[ltf[letter]].append(letter)

    for freq in ftl:
        ftl[freq].sort(key=sortlet.find, reverse=True)
        ftl[freq] = ''.join(ftl[freq])

    freqPairs = list(ftl.items())
    freqPairs.sort(key=getkey, reverse=True)
    freqOrder = []

    for freqPair in freqPairs:
        freqOrder.append(freqPair[1])

    return ''.join(freqOrder)


# In[3]:


with open("in.txt") as fl:
    for line in fl:
        print(line)


# In[4]:


mes = line
getcount(mes)


# In[5]:


ansfre = getorder(mes)
ansfre


# In[6]:


sortlet


# In[7]:


z = []
def permutations(start, end=[]):
    if len(start) == 0:
        z.append(end)
#         print(end)
    else:
        for i in range(len(start)):
            permutations(start[:i] + start[i+1:], end + start[i:i+1])
            
# permutations(sortlet)#得到窮舉解
sol = []
for item in z:
    dictionary = dict(zip(ansfre,item))#得到每組英文與密文對照字典
    print(dictionary)
    
    line = []
    for items in mes:#對照字典翻譯密文
        c = dictionary[items]
        line.append(c)
    line = ''.join(line).lower()
    print(line)
    sol.append(line)#各組解的列表
    print(sol)


# In[8]:


dictionary = dict(zip(ansfre,sortlet))
dictionary


# In[9]:


# data = []
# for key in keymap:#先找u對etaoin的keymap解密
    line = []
    for item in mes:
        c = dictionary[item]
        line.append(c)
    line = ''.join(line).lower()
    print(line)


# In[10]:


lettertonum = {'A': 0, 'B': 1, 'C': 2, 'D':3 , 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13, 
                     'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25}


# In[11]:


numtoletter = {0:'A', 1:'B', 2:'C', 3:'D' , 4:'E',  5:'F',  6:'G',  7:'H',  8:'I',  9:'J',  10:'K',  11:'L',  12:'M',  13:'N', 
                     14: 'O',  15:'P',  16:'Q',  17:'R',  18:'S',  19:'T', 20: 'U', 21: 'V',  22:'W',  23:'X',  24:'Y',  25:'Z'}


# In[12]:


lettertonum['U']-lettertonum['E']


# In[13]:


numtoletter[20]


# In[14]:


map = []
for item in mes:
    c = numtoletter[(lettertonum[item]-20)%26]
    map.append(c)
map = ''.join(map)
print(map)


# In[15]:


keymap = []
for item in sortlet:
    key = lettertonum[ansfre[0]]-lettertonum[item]
    keymap.append(key)
keymap


# In[16]:


# import pandas as pd
# from pandas import dataFrame 
data = []
for key in keymap:#先找u對etaoin的keymap解密
    guess = []
    for item in mes:
        c = numtoletter[(lettertonum[item]-key)%26]
        guess.append(c)
    guess = ''.join(guess).lower()
    data.append(guess)
    print(guess)


# In[17]:


import pandas as pd
from pandas import DataFrame
code = DataFrame(data)
code


# In[18]:


import flair
sentiment_model = flair.models.TextClassifier.load('en-sentiment')


# In[19]:


import re
stop = []
sentiments = []

whitespace = re.compile(r"a+")
web_address = re.compile('http:\/\/[a-z0-9.~_\-\/]+')
user = re.compile(r"(?i)@[a-z0-9_]+")

tweets = code
for tweet in tweets[0]:
    # we then use the sub method to replace anything matching
        tweet = whitespace.sub(' a ', tweet)
            # print(tweet)
        tweet = web_address.sub('', tweet)
        tweet = user.sub('', tweet)
        sentence = flair.data.Sentence(tweet)
        sentiment_model.predict(sentence)
    # extract sentiment prediction
        sentiments.append(sentence.labels[0])  # 'POSITIVE' or 'NEGATIVE'
        stop.append(tweet)  # 'POSITIVE' or 'NEGATIVE'
        print(tweet)
tweets['stop'] = stop
tweets['sentiment'] = sentiments

tweets


# In[20]:


# import pandas as pd
# data = pd.read_csv("spam1.csv", encoding='latin-1').sample(frac=1).drop_duplicates()

# data = data[['v1', 'v2']].rename(columns={"v1":"label", "v2":"text"})

# data['label'] = '__label__' + data['label'].astype(str)

# data.iloc[0:int(len(data)*0.8)].to_csv('train.csv', sep='\t', index = False, header = False)
# data.iloc[int(len(data)*0.8):int(len(data)*0.9)].to_csv('test.csv', sep='\t', index = False, header = False)
# data.iloc[int(len(data)*0.9):].to_csv('dev.csv', sep='\t', index = False, header = False);


# In[21]:


# from flair.data import Corpus
# from flair.datasets import ClassificationCorpus

# # this is the folder in which train, test and dev files reside
# data_folder = './'

# # load corpus containing training, test and dev data
# corpus = ClassificationCorpus(data_folder,
#                                       test_file='test.csv',
#                                       dev_file='dev.csv',
#                                       train_file='train.csv',                                       
#                                       label_type='topic'
#                                       )


# In[22]:


# from flair.data_fetcher import NLPTaskDataFetcher
# from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentLSTMEmbeddings
# from flair.models import TextClassifier
# from flair.trainers import ModelTrainer
# from pathlib import Path
# from flair.data import Corpus
# from flair.datasets import CSVClassificationCorpus

# # column_name_map = {4: "text", 1: "label_topic", 2: "label_subtopic"}
# # corpus = CSVClassificationCorpus(Path('./'), column_name_map, test_file='test.csv', dev_file='dev.csv', train_file='train.csv')
# corpus.make_label_dictionary(label_type='topic')

# word_embeddings = [WordEmbeddings('glove'), FlairEmbeddings('news-forward-fast'), FlairEmbeddings('news-backward-fast')]

# document_embeddings = DocumentLSTMEmbeddings(word_embeddings, hidden_size=512, reproject_words=True, reproject_words_dimension=256)

# classifier = TextClassifier(document_embeddings, label_dictionary=corpus.make_label_dictionary(label_type='topic'),label_type='topic', multi_label=False)

# trainer = ModelTrainer(classifier, corpus)

# trainer.train('./', max_epochs=5)


# In[23]:


from flair.models import TextClassifier
from flair.data import Sentence

classifier = TextClassifier.load('./best-model.pt')

sentence = Sentence('aprogramcracks a substitutioncipher')

classifier.predict(sentence)

# print(sentence.labels)


# In[24]:


val = []
for item in code['stop']:
    sentence = Sentence(item)

    classifier.predict(sentence)
    print(item)
    print(sentence.labels)
    val.append(sentence.labels)
code['validation'] = val
code


# In[25]:


freq = []
# eee = df[df['checke']==0]
for item in code[0]:
    cnt = 0
    for c in item:
        if (c=='e'):
            cnt = cnt+1 
        if (c=='t'):
            cnt = cnt+1 
        if (c=='a'):
            cnt = cnt+1 
        if (c=='o'):
            cnt = cnt+1 
        if (c=='i'):
            cnt = cnt+1  
        if (c=='n'):
            cnt = cnt+1
        if (c=='z'):
            cnt = cnt-1
        if (c=='q'):
            cnt = cnt-1
        if (c=='x'):
            cnt = cnt-1
        if (c=='j'):
            cnt = cnt-1
        if (c=='k'):
            cnt = cnt-1
        if (c=='v'):
            cnt = cnt-1
    print(item)
    freq.append(cnt)
code['freq'] = freq
code


# In[26]:


import pandas as pd
raw = pd.read_csv('dataset.csv')
raw = raw[raw['language']!='Latin']
# print(raw)
languages = set(raw['language'])


# In[27]:


import numpy as np
from sklearn.model_selection import train_test_split

X=raw['Text']
y=raw['language']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print(len(X_train))
# print(len(X_test))
# print(len(y_train))
# print(len(y_test))


# In[28]:


X_test


# In[29]:


c_test = code[0]
print(c_test)


# In[30]:


# Extract Unigrams
from sklearn.feature_extraction.text import CountVectorizer
unigramVectorizer = CountVectorizer(analyzer='char', ngram_range=(1,1))
X_unigram_train_raw = unigramVectorizer.fit_transform(X_train)
X_unigram_test_raw = unigramVectorizer.transform(X_test)
c_unigram_test_raw = unigramVectorizer.transform(c_test)


unigramFeatures = unigramVectorizer.get_feature_names()

# print('Number of unigrams in training set:', len(unigramFeatures))


# In[31]:


def train_lang_dict(X_raw_counts, y_train):
    lang_dict = {}
    for i in range(len(y_train)):
        lang = y_train[i]
        v = np.array(X_raw_counts[i])
        if not lang in lang_dict:
            lang_dict[lang] = v
        else:
            lang_dict[lang] += v
            
    # to relative
    for lang in lang_dict:
        v = lang_dict[lang]
        lang_dict[lang] = v / np.sum(v)
        
    return lang_dict

language_dict_unigram = train_lang_dict(X_unigram_train_raw.toarray(), y_train.values)

# Collect relevant chars per language
def getRelevantCharsPerLanguage(features, language_dict, significance=1e-5):
    relevantCharsPerLanguage = {}
    for lang in languages:
        chars = []
        relevantCharsPerLanguage[lang] = chars
        v = language_dict[lang]
        for i in range(len(v)):
            if v[i] > significance:
                chars.append(features[i])
    return relevantCharsPerLanguage

relevantCharsPerLanguage = getRelevantCharsPerLanguage(unigramFeatures, language_dict_unigram)
    
# Print number of unigrams per language
for lang in languages:    
    print(lang, len(relevantCharsPerLanguage[lang]))


# In[32]:


# get most common chars for a few European languages
europeanLanguages = ['Portugese', 'Spanish', 'English', 'Dutch', 'Swedish']
relevantChars_OnePercent = getRelevantCharsPerLanguage(unigramFeatures, language_dict_unigram, 1e-2)

# collect and sort chars
europeanCharacters = []
for lang in europeanLanguages:
    europeanCharacters += relevantChars_OnePercent[lang]
europeanCharacters = list(set(europeanCharacters))
europeanCharacters.sort()

# build data
indices = [unigramFeatures.index(f) for f in europeanCharacters]
data = []
for lang in europeanLanguages:
    data.append(language_dict_unigram[lang][indices])

#build dataframe
df = pd.DataFrame(np.array(data).T, columns=europeanLanguages, index=europeanCharacters)
df.index.name = 'Characters'
df.columns.name = 'Languages'


# In[33]:


# number of bigrams
from sklearn.feature_extraction.text import CountVectorizer
bigramVectorizer = CountVectorizer(analyzer='char', ngram_range=(2,2))
X_bigram_raw = bigramVectorizer.fit_transform(X_train)
bigramFeatures = bigramVectorizer.get_feature_names()
print('Number of bigrams', len(bigramFeatures))


# In[34]:


# top bigrams (>1%) for Spanish, Italian (Latin), English, Dutch, Chinese, Japanese, Korean
language_dict_bigram = train_lang_dict(X_bigram_raw.toarray(), y_train.values)
relevantCharsPerLanguage = getRelevantCharsPerLanguage(bigramFeatures, language_dict_bigram, significance=1e-2)
print('English', relevantCharsPerLanguage['English'])


# In[35]:


# Uni- & Bi-Gram Mixture CountVectorizer for top 1% features
from sklearn.feature_extraction.text import CountVectorizer

top1PrecentMixtureVectorizer = CountVectorizer(analyzer='char', ngram_range=(1,2), min_df=1e-2)
X_top1Percent_train_raw = top1PrecentMixtureVectorizer.fit_transform(X_train)
X_top1Percent_test_raw = top1PrecentMixtureVectorizer.transform(X_test)
c_top1Percent_test_raw = top1PrecentMixtureVectorizer.transform(c_test)

language_dict_top1Percent = train_lang_dict(X_top1Percent_train_raw.toarray(), y_train.values)

top1PercentFeatures = top1PrecentMixtureVectorizer.get_feature_names()
# print('Length of features', len(top1PercentFeatures))
# print('')

#Unique features per language
relevantChars_Top1Percent = getRelevantCharsPerLanguage(top1PercentFeatures, language_dict_top1Percent, 1e-5)
for lang in relevantChars_Top1Percent:
    print("{}: {}".format(lang, len(relevantChars_Top1Percent[lang])))


# In[36]:


def getRelevantGramsPerLanguage(features, language_dict, top=50):
    relevantGramsPerLanguage = {}
    for lang in languages:
        chars = []
        relevantGramsPerLanguage[lang] = chars
        v = language_dict[lang]
        sortIndex = (-v).argsort()[:top]
        for i in range(len(sortIndex)):
            chars.append(features[sortIndex[i]])
    return relevantGramsPerLanguage

top50PerLanguage_dict = getRelevantGramsPerLanguage(top1PercentFeatures, language_dict_top1Percent)

# top50
allTop50 = []
for lang in top50PerLanguage_dict:
    allTop50 += set(top50PerLanguage_dict[lang])

top50 = list(set(allTop50))
    
print('All items:', len(allTop50))
print('Unique items:', len(top50))


# In[37]:


# getRelevantColumnIndices
def getRelevantColumnIndices(allFeatures, selectedFeatures):
    relevantColumns = []
    for feature in selectedFeatures:
        relevantColumns = np.append(relevantColumns, np.where(allFeatures==feature))
    return relevantColumns.astype(int)

relevantColumnIndices = getRelevantColumnIndices(np.array(top1PercentFeatures), top50)


X_top50_train_raw = np.array(X_top1Percent_train_raw.toarray()[:,relevantColumnIndices])
X_top50_test_raw = X_top1Percent_test_raw.toarray()[:,relevantColumnIndices] 
c_top50_test_raw = c_top1Percent_test_raw.toarray()[:,relevantColumnIndices] 

# print('train shape', X_top50_train_raw.shape)
# print('test shape', X_top50_test_raw.shape)
# print('c shape', c_top50_test_raw.shape)


# In[38]:


# Define some functions for our purpose

from sklearn.preprocessing import normalize
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sn
import matplotlib.pyplot as plt
import scipy

# Utils for conversion of different sources into numpy array
def toNumpyArray(data):
    data_type = type(data)
    if data_type == np.ndarray:
        return data
    elif data_type == list:
        return np.array(data_type)
    elif data_type == scipy.sparse.csr.csr_matrix:
        return data.toarray()
#     print(data_type)
    return None


def normalizeData(train, test):
    train_result = normalize(train, norm='l2', axis=1, copy=True, return_norm=False)
    test_result = normalize(test, norm='l2', axis=1, copy=True, return_norm=False)
    return train_result, test_result

def applyNaiveBayes(X_train, y_train, X_test):
    trainArray = toNumpyArray(X_train)
    testArray = toNumpyArray(X_test)
    
    clf = MultinomialNB()
    clf.fit(trainArray, y_train)
    y_predict = clf.predict(testArray)
    return y_predict

def plot_F_Scores(y_test, y_predict):
    f1_micro = f1_score(y_test, y_predict, average='micro')
    f1_macro = f1_score(y_test, y_predict, average='macro')
    f1_weighted = f1_score(y_test, y_predict, average='weighted')
#     print("F1: {} (micro), {} (macro), {} (weighted)".format(f1_micro, f1_macro, f1_weighted))
    
def plot_Confusion_Matrix(y_test, y_predict, color="Blues"):
    allLabels = list(set(list(y_test) + list(y_predict)))
    allLabels.sort()
    confusionMatrix = confusion_matrix(y_test, y_predict, labels=allLabels)
    unqiueLabel = np.unique(allLabels)
    df_cm = pd.DataFrame(confusionMatrix, columns=unqiueLabel, index=unqiueLabel)
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'

    sn.set(font_scale=0.8) # for label size
    sn.set(rc={'figure.figsize':(15, 15)})
    sn.heatmap(df_cm, cmap=color, annot=True, annot_kws={"size": 12}, fmt='g')# font size
    plt.show()


# In[39]:


X_top50_train, X_top50_test = normalizeData(X_top50_train_raw, X_top50_test_raw)
trainArray = toNumpyArray(X_top50_train)
testArray = toNumpyArray(X_top50_test)
    
clf = MultinomialNB()
clf.fit(trainArray, y_train)
y_predict = clf.predict(testArray)


# In[40]:


c_top50_test = normalize(c_top50_test_raw, norm='l2', axis=1, copy=True, return_norm=False)
ctestArray = toNumpyArray(c_top50_test)

c_predict = clf.predict(ctestArray)


# In[41]:


ans = DataFrame(c_predict)
ans


# In[42]:


code['language'] = ans
code


# In[43]:


english = code[code['language']=='English']
english


# In[44]:


sort = english.sort_values(["freq"], ascending=False)
sort


# In[45]:


for i in sort[0]:
    print(i)


# In[46]:


out = sort.head(1)
out


# In[47]:


#print
for item in out[0]:
    print(item)


# In[48]:


path = 'out.txt'
data=open(path,'w+') 
for item in out[0]:
        print(item,file=data)
data.close()


# In[ ]:




