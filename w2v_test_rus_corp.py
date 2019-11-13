from gensim.models import KeyedVectors
from nltk import pos_tag, word_tokenize

model = KeyedVectors.load_word2vec_format("ruscorpora_1_300_10.bin", binary=True)

text = word_tokenize("кратко опишем алгоритм")
textTag = pos_tag(text, tagset='universal')
tagged = [t[0] + "_" + t[1] for t in textTag]
print(tagged)

for n in model.most_similar(tagged):
    print(str(n[0]) + " " + str(n[1]))

# print(model[u'пожар_NOUN'])