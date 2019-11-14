import re
from string import punctuation

import numpy as np
from gensim.models import KeyedVectors
from nltk import pos_tag, word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus.reader import wordnet
from pymystem3 import Mystem

mystem = Mystem()
russian_stopwords = stopwords.words("russian")


def preprocess_text(text):
    tokens = mystem.lemmatize(text.lower())
    tokens = [token for token in tokens if token not in russian_stopwords \
              and token != " " \
              and token.strip() not in punctuation]

    return tokens


def find_words(word):
    return [i for i in model.index2word if i.startswith(word + "_")]


tag_dict = {"J": wordnet.ADJ_SAT,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV}

model = KeyedVectors.load_word2vec_format("ruscorpora_1_300_10.bin", binary=True)
lemmatizer = WordNetLemmatizer()


def w2v_get_vec(sentence):
    textTag = pos_tag(preprocess_text("кратко краток краткий"), tagset='universal')
    tagged = [t[0] + "_" + t[1] for t in textTag]
    word_vecs = []
    for i in textTag:
        t = i[0] + "_" + i[1]
        if t in model.vocab:
            word_vecs.append(model[t])
            # print(t + " in vocabulary")
        else:
            similar_words = find_words(i[0])
            if len(similar_words) > 0:
                word_vecs.append(model[similar_words[0]])
                # print(i[0] + " form in vocabulary")
            else:
                word_vecs.append(np.asarray([0] * model.vector_size))
                # print(i[0] + " not found")

    return sum(word_vecs)

    # text = preprocess_text("кратко краток краткий")
# textTag = pos_tag(preprocess_text("кратко краток краткий"), tagset='universal')
# tagged = [t[0] + "_" + t[1] for t in textTag]
# print(tagged)

# for n in model.most_similar(tagged):
#     print(str(n[0]) + " " + str(n[1]))


# print(model[u'пожар_NOUN'])