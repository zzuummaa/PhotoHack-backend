import json
import time

import numpy as np
import spacy
from gensim.models import KeyedVectors
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus.reader import wordnet

from nlp_tools import load_training_pairs

nlp = spacy.load('ru2', disable=['parser', 'NER'])
nlp.add_pipe(nlp.create_pipe('sentencizer'), first=True)

russian_stopwords = stopwords.words("russian")


def preprocess_text(text):
    doc = nlp(text)
    lemma = []
    for s in doc.sents:
        for t in s:
            lemma.append((t.lemma_, t.pos_))

    return lemma


def find_words(word):
    return [i for i in model.index2word if i.startswith(word + "_")]


tag_dict = {"J": wordnet.ADJ_SAT,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV}

model = KeyedVectors.load_word2vec_format("ruscorpora_1_300_10.bin", binary=True)
lemmatizer = WordNetLemmatizer()


def w2v_get_vec(sentence):
    textTag = preprocess_text(sentence)
    if len(textTag) == 0:
        return np.asarray([0] * model.vector_size)
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

    return sum(word_vecs) / len(textTag)

    # text = preprocess_text("кратко краток краткий")


# textTag = pos_tag(preprocess_text("кратко краток краткий"), tagset='universal')
# tagged = [t[0] + "_" + t[1] for t in textTag]
# print(tagged)

# for n in model.most_similar(tagged):
#     print(str(n[0]) + " " + str(n[1]))


# print(model[u'пожар_NOUN'])

if __name__ == '__main__':
    sentences, target = load_training_pairs("trainingPairs.json")
    y = np.asarray(target)

    # X = np.zeros((len(sentences), len(mydict)))
    # i = 0
    X = np.zeros((len(sentences), model.vector_size))
    last_time = time.time()
    last_i = 0
    for i in range(len(sentences)):
        X[i] = w2v_get_vec(sentences[i])

        if time.time() - last_time > 3:
            cur_time = time.time()
            speed = (i - last_i) / (cur_time - last_time)
            last_i = i
            last_time = cur_time
            print(str(i) + "/" + str(len(sentences)) + " " + ("%.2f" % round(speed, 2)) + " op/sec")

    print(str(len(sentences)) + "/" + str(len(sentences)) + " - completed")

    with open("w2vVectors.json", 'w', encoding='utf-8') as f:
        json.dump(list(zip(sentences, X.tolist(), target)), f, ensure_ascii=False)
