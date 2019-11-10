import string
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

inStr = "привет, мне тут передали у тебя проблемы с апи были? Судя по скрину тебе айди шаблона вернулся, но в сам скрин не влезло, разобрался?"

tokens = word_tokenize(inStr)
print("tokenized:" + str(tokens))

tokens = [i for i in tokens if (i not in string.punctuation)]
print("removed punctuation:" + str(tokens))

stop_words = stopwords.words('russian')
stop_words.extend(['что', 'это', 'так', 'вот', 'быть', 'как', 'в', '—', '–', 'к', 'на', '...'])
tokens = [i for i in tokens if (i not in stop_words)]
print("removed stop words:" + str(tokens))

digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
tokens = [i for i in tokens if (i)]

stemmer = SnowballStemmer("russian")

outStr = ""
i = 0
outWords = list()
for token in tokens:
    outWords.append(stemmer.stem(token))
    outStr = outStr + " " + outWords[-1]

print("out words:" + str(outWords))
print(outStr)
