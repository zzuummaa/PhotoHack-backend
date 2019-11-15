import json

import sys
import string
import numpy as np
from gensim import corpora
from nltk import SnowballStemmer, word_tokenize
from nltk.corpus import stopwords

print(__doc__)

try:
    fileName = 'one-hot_encoding.dict'
    mydict = corpora.Dictionary.load(fileName)
    fileName = "trainingPairs.json"
    jsonData = json.load(open(fileName, encoding='utf-8', newline=''))
except FileNotFoundError:
    sys.stderr.write("Warn: file " + fileName + " not found")

stop_words = stopwords.words('russian')
stemmer = SnowballStemmer("russian")

template_ids = [0, 1001996, 1001997, 0, 0, 0, 0]
target_ids = [0, 3, 4, 5, 7, 9, 10]
target_names = ["Хз", "Отпуск", "Работа", "Проект, планировани", "Работа с компьютером", "Ура", "Нет"]


def prepare_sentence(sentence):
    tokens = word_tokenize(sentence.lower())
    tokens = [i for i in tokens if (i not in string.punctuation)]
    # tokens = [i for i in tokens if (i not in stop_words)]

    # TODO should just remove digits, but not words
    digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    tokens = [i for i in tokens if not i in digits]

    return tokens


def extract_fitures(words):
    fitures = np.asarray([0] * len(mydict))
    for word in words:
        if word in mydict.token2id:
            j = mydict.token2id[word]
            fitures[j] = fitures[j] + 1

    return fitures


def load_training_set():
    sentences = [i[0] for i in jsonData]
    target = [i[1][0] for i in jsonData]

    y = np.asarray(target)

    X = np.zeros((len(sentences), len(mydict.token2id)))
    i = 0
    for sentence in sentences:
        words = sentence.split()
        for word in words:
            if word in mydict.token2id:
                j = mydict.token2id[word]
                X[i][j] = X[i][j] + 1
        i = i + 1

    return X, y


def load_training_pairs(filename):
    jsonData = json.load(open(filename, encoding='utf-8', newline=''))
    sentences = [i[0] for i in jsonData]
    target = [i[1][0] for i in jsonData]
    return sentences, target
