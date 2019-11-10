from gensim import models
from gensim import corpora
from gensim.utils import simple_preprocess
import numpy as np

documents = ["This is the first line",
             "This is the second sentence",
             "This third document"]
# Create the Dictionary and Corpus
mydict = corpora.Dictionary([simple_preprocess(line) for line in documents])
corpus = [mydict.doc2bow(simple_preprocess(line)) for line in documents]
# Show the Word Weights in Corpus
for doc in corpus:
    print([[mydict[id], freq] for id, freq in doc])
# [['first', 1], ['is', 1], ['line', 1], ['the', 1], ['this', 1]]
# [['is', 1], ['the', 1], ['this', 1], ['second', 1], ['sentence', 1]]
# [['this', 1], ['document', 1], ['third', 1]]
# Create the TF-IDF model
tfidf = models.TfidfModel(corpus, smartirs='ntc')
# Show the TF-IDF weights
for doc in tfidf[corpus]:
    print([[mydict[id], np.around(freq, decimals=2)] for id, freq in doc])
# [['first', 0.66], ['is', 0.24], ['line', 0.66], ['the', 0.24]]
# [['is', 0.24], ['the', 0.24], ['second', 0.66], ['sentence', 0.66]]
# [['document', 0.71], ['third', 0.71]]