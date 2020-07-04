# tfidf-neural

### The purpose of this project is to take the dataset [`OnionOrNot.csv`](https://www.kaggle.com/chrisfilo/onion-or-not?select=OnionOrNot.csv) and perform the following operations:

- Split the dataset into 75% and 25% segments (training & testing data)
- Clear up the sentences (remove stopwords, POS tag, lemmatize) with NLTK
- Create TF-IDF vectors from the sentences
- Configure & train a neural net with the tfidf vectors of the training dataset
- Check percision, recall and f1 score with the testing data

# Installation instructions

You will need python 3 for this project. It is recomended that you use `pyenv` for this. Installation instructions for `pyenv` can be found [here](https://github.com/pyenv/pyenv).
Once you have `pyenv` installed you need to get `pipenv` for the specified python version. To do that run:

```
pip install pipenv
```

Then you need to clone the project:

```
git clone git@github.com:thanosstamatakis/tfidf-neural.git
```

After that run the following to get the project dependencies:

```
cd tfidf-neural/
pipenv install
```

Once this is done setup plaid-ml with the following command (to enable GPU acceleration):

```
plaidml-setup
```

Once this is done, you can just run the project:

```
pipenv shell
python script.py
```
