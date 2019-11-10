import json

import numpy as np
from gensim import corpora

print(__doc__)

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

mydict = corpora.Dictionary.load('one-hot_encoding.dict')
mydict = mydict.token2id
jsonData = json.load(open("trainingPairs.json", encoding='utf-8', newline=''))
sentences = [i[0] for i in jsonData]
target = [i[1][0] for i in jsonData]

y = np.asarray(target)

X = np.zeros((len(sentences), len(mydict)))
i = 0
for sentence in sentences:
    words = sentence.split()
    for word in words:
        if word in mydict:
            j = mydict[word]
            X[i][j] = X[i][j] + 1
    i = i + 1

target_names = ["Отпуск", "Работа", "Проект, планировани", "Работа с компьютером", "Ура", "Нет"]

pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit(X, y).transform(X)

# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

plt.figure()
colors = ['navy', 'turquoise', 'darkorange', 'brown', 'green', "black"]
lw = 2

for color, i, target_name in zip(colors, [3, 4, 5, 7, 9, 10], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of IRIS dataset')

plt.figure()
for color, i, target_name in zip(colors, [3, 4, 5, 7, 9, 10], target_names):
    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of IRIS dataset')

plt.show()
