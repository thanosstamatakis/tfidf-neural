import string
import pandas as pd
import nltk
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.stem import PorterStemmer
# Used for pre-processing data

def preprocessing(text):
    text2 = " ".join("".join([" " if ch in string.punctuation else ch for ch in text]).split())
    tokens = [word for sent in nltk.sent_tokenize(text2) for word in nltk.word_tokenize(sent)]
    tokens = [word.lower() for word in tokens]
    stopwds = stopwords.words('english')
    tokens = [token for token in tokens if token not in stopwds]
    tokens = [word for word in tokens if len(word)>=3]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    tagged_corpus = pos_tag(tokens)
    pre_proc_text = " ".join([prat_lemmatize(token,tag) for token,tag in tagged_corpus])
    return pre_proc_text

def prat_lemmatize(token,tag):

    Noun_tags = ['NN','NNP','NNPS','NNS']
    Verb_tags = ['VB','VBD','VBG','VBN','VBP','VBZ']
    lemmatizer = WordNetLemmatizer()

    if tag in Noun_tags:
        return lemmatizer.lemmatize(token,'n')
    elif tag in Verb_tags:
        return lemmatizer.lemmatize(token,'v')
    else:
        return lemmatizer.lemmatize(token,'n')

newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')
x_train = newsgroups_train.data
x_test = newsgroups_test.data
y_train = newsgroups_train.target
y_test = newsgroups_test.target
# print (newsgroups_train.target_names)
# print ("n")
# print ("Sample Email:")
print (x_train[0])
# print ("Sample Target Category:")
print (y_train[0])
# print (newsgroups_train.target_names[y_train[0]])
print(len(x_train))
x_train_preprocessed = []
for i in x_train:
    x_train_preprocessed.append(preprocessing(i))

print("Done with x-train...")

print(len(x_test))
x_test_preprocessed = []
for i in x_test:
    x_test_preprocessed.append(preprocessing(i))

print("Done with x-test...")