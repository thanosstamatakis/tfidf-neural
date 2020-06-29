import string
import nltk
import csv
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adadelta, Adam, RMSprop
from keras.utils import np_utils
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.stem import PorterStemmer


def preprocessing(text):
    text2 = " ".join(
        "".join([" " if ch in string.punctuation else ch for ch in text]).split())
    tokens = [word for sent in nltk.sent_tokenize(
        text2) for word in nltk.word_tokenize(sent)]
    tokens = [word.lower() for word in tokens]
    stopwds = stopwords.words('english')
    tokens = [token for token in tokens if token not in stopwds]
    tokens = [word for word in tokens if len(word) >= 3]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    tagged_corpus = pos_tag(tokens)
    pre_proc_text = " ".join([prat_lemmatize(token, tag)
                              for token, tag in tagged_corpus])
    return pre_proc_text


def prat_lemmatize(token, tag):

    Noun_tags = ['NN', 'NNP', 'NNPS', 'NNS']
    Verb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    lemmatizer = WordNetLemmatizer()

    if tag in Noun_tags:
        return lemmatizer.lemmatize(token, 'n')
    elif tag in Verb_tags:
        return lemmatizer.lemmatize(token, 'v')
    else:
        return lemmatizer.lemmatize(token, 'n')


with open('file.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    x = []
    y = []

    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            x.append(row[0])
            y.append(int(row[1]))

# x = ['thanos', 'markos', 'giorgos', 'bill']
# y = [0, 1, 2, 3]

# Calculate 75% of array to get training dataset
sfpc = int(len(x)*0.75)

x = np.array(x)
y = np.array(y)

x_train = x[0:sfpc]
x_test = x[sfpc:]
y_train = y[0:sfpc]
y_test = y[sfpc:]

print(len(x_train))
x_train_preprocessed = []
for i in x_train:
    x_train_preprocessed.append(preprocessing(i))

print("Done with x-train...")
print(x_train_preprocessed[0])
print(x_train_preprocessed[1])
print(x_train_preprocessed[2])

print(len(x_test))
x_test_preprocessed = []
for i in x_test:
    x_test_preprocessed.append(preprocessing(i))

print("Done with x-test...")

vectorizer = TfidfVectorizer(min_df=2, ngram_range=(
    1, 2), stop_words='english', max_features=10000, strip_accents='unicode', norm='l2')
x_train_2 = vectorizer.fit_transform(x_train_preprocessed).todense()
x_test_2 = vectorizer.transform(x_test_preprocessed).todense()

print(x_train_2)

np.random.seed(1337)
nb_classes = 2
batch_size = 64
nb_epochs = 20

Y_train = np_utils.to_categorical(y_train, nb_classes)

# Deep Layer Model building in Keras

#del model

model = Sequential()
model.add(Dense(1000, input_shape=(10000,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
print(model.summary())

model.fit(x_train_2, Y_train, batch_size=batch_size,
          epochs=nb_epochs, verbose=1)
