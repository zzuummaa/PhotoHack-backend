import json
from gensim import corpora, models
from gensim.utils import simple_preprocess
from nlp_tools import prepare_sentence, stemmer

trainingSet = json.load(open("trainingSet.json", encoding='utf-8', newline=''))

messages = list()
targets = list()
dictMessages = list()
dictTargets = list()
for example in trainingSet:
    dictMessages.append(example["text"])
    dictTargets.append(example["target"])
    if len(example["target"]) > 0:
        messages.append(example["text"])
        targets.append(example["target"])


def prepare_mesages(messages, fileName):
    # Tokenizing, remove punctuation, stemming
    sentences = list()
    for message in messages:
        tokens = prepare_sentence(message)
        if len(tokens) > 0:
            sentence = ""
            for token in tokens:
                sentence = sentence + " " + stemmer.stem(token)

            sentences.append(sentence)

    trainingPairs = list(zip(sentences, targets))
    with open(fileName, 'w', encoding='utf-8') as f:
        json.dump(trainingPairs, f, ensure_ascii=False)

    return sentences


sentences = prepare_mesages(messages, "trainingPairs.json")
dictSentences = prepare_mesages(dictMessages, "dictPairs.json")

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
