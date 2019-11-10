import gensim
import string
from gensim import corpora, models
from gensim.utils import simple_preprocess
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import json

trainingSet = json.load(open("trainingSet.json", encoding='utf-8', newline=''))

messages = list()
targets = list()
for example in trainingSet:
    if len(example["target"]) > 0:
        messages.append(example["text"])
        targets.append(example["target"])

stop_words = stopwords.words('russian')
stemmer = SnowballStemmer("russian")

# Tokenizing, remove punctuation, stemming
sentences = list()
for message in messages:
    tokens = word_tokenize(message)
    tokens = [i for i in tokens if (i not in string.punctuation)]

    tokens = [i for i in tokens if (i not in stop_words)]

    digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    tokens = [i for i in tokens if i]

    if len(tokens) > 0:
        sentence = ""
        for token in tokens:
            sentence = sentence + " " + stemmer.stem(token)

        sentences.append(sentence)

trainingPairs = list(zip(sentences, targets))
with open("trainingPairs.json", 'w', encoding='utf-8') as f:
    json.dump(trainingPairs, f, ensure_ascii=False)

# Create the Dictionary and Corpus
mydict = corpora.Dictionary([simple_preprocess(line) for line in sentences])
corpus = [mydict.doc2bow(simple_preprocess(line)) for line in sentences]
# Show the Word Weights in Corpus
# for doc in corpus:
#     print([[mydict[id], freq] for id, freq in doc])

tfidf = models.TfidfModel(corpus, smartirs='ntc')
# Show the TF-IDF weights
# for doc in tfidf[corpus]:
#     print([[mydict[id], np.around(freq, decimals=2)] for id, freq in doc])

mydict.save("one-hot_encoding.dict")