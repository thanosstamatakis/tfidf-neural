import string
import nltk
import csv
import pandas as pd
import numpy as np
import plaidml.keras
import os
plaidml.keras.install_backend()
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
from sklearn.metrics import accuracy_score, classification_report
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


def nltk_preprocess(text):
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
    pre_proc_text = " ".join([nltk_lematize(token, tag)
                              for token, tag in tagged_corpus])
    return pre_proc_text


def nltk_lematize(token, tag):

    Noun_tags = ['NN', 'NNP', 'NNPS', 'NNS']
    Verb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    lemmatizer = WordNetLemmatizer()

    if tag in Noun_tags:
        return lemmatizer.lemmatize(token, 'n')
    elif tag in Verb_tags:
        return lemmatizer.lemmatize(token, 'v')
    else:
        return lemmatizer.lemmatize(token, 'n')

# Read csv and get sentences and expected outputs
with open('file.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    sentences = []
    expected_outputs = []

    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            sentences.append(row[0])
            expected_outputs.append(int(row[1]))


# Calculate 75% of array to get training dataset
sfpc = int(len(sentences)*0.75)

sentences = np.array(sentences)
expected_outputs = np.array(expected_outputs)

# Split dataset 75%(training)-25%(testing)
training_dataset = sentences[0:sfpc]
testing_dataset = sentences[sfpc:]
training_expected_output = expected_outputs[0:sfpc]
testing_expected_output = expected_outputs[sfpc:]

# Preprocess training dataset using nltk
print(f"Length of training dataset: {len(training_dataset)} documents")
preprocessed_training_dataset = []
for i in training_dataset:
    preprocessed_training_dataset.append(nltk_preprocess(i))
print("Done preprocessing training dataset...")


# Preprocess testing dataset using nltk
print(f"Length of testing dataset: {len(testing_dataset)} documents")
preprocessed_testing_dataset = []
for i in testing_dataset:
    preprocessed_testing_dataset.append(nltk_preprocess(i))
print("Done preprocessing testing dataset...")


vectorizer = TfidfVectorizer(min_df=2, ngram_range=(
    1, 2), stop_words='english', strip_accents='unicode', norm='l2')
tfidf_training_dataset = vectorizer.fit_transform(preprocessed_training_dataset).todense()
tfidf_testing_dataset = vectorizer.transform(preprocessed_testing_dataset).todense()

print(f"Training vector size: {tfidf_training_dataset.shape}")
print(f"Testing vector size: {tfidf_testing_dataset.shape}")


# Net configuration
np.random.seed(1337)
num_of_classes = 2
batch_size = 64
num_of_epochs = 20

# One hot encode the output classes
one_hot_encoded_cats = np_utils.to_categorical(training_expected_output, num_of_classes)

# Deep Layer Model in Keras
model = Sequential()
model.add(Dense(1000, input_shape=(tfidf_training_dataset.shape[1],)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_of_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
print(model.summary())

# Train model
model.fit(tfidf_training_dataset, one_hot_encoded_cats, batch_size=batch_size,
          epochs=num_of_epochs, verbose=1)

training_dataset_predclass = model.predict_classes(tfidf_training_dataset, batch_size=batch_size)
testing_dataset_predclass = model.predict_classes(tfidf_testing_dataset, batch_size=batch_size)

# Print results
print(f"OnionOrNot Neural Network - Train accuracy: {(round(accuracy_score(training_expected_output, training_dataset_predclass), 3))}")
print(f"OnionOrNot Neural Network - Test accuracy: {(round(accuracy_score(testing_expected_output, testing_dataset_predclass), 3))}")
print("OnionOrNot Neural Network - Train Classification Report")
print(classification_report(training_expected_output, training_dataset_predclass))
print("OnionOrNot Neural Network - Test Classification Report")
print(classification_report(testing_expected_output, testing_dataset_predclass))
