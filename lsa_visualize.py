import math

import matplotlib
import matplotlib.patches as mpatches
import numpy as np
from sklearn.decomposition import TruncatedSVD

from nlp_tools import target_names, load_training_pairs
from w2v_test_rus_corp import w2v_get_vec, model

print(__doc__)

import matplotlib.pyplot as plt

sentences, target = load_training_pairs("trainingPairs.json")
y = np.asarray(target)

# X = np.zeros((len(sentences), len(mydict)))
# i = 0
X = np.zeros((len(sentences), model.vector_size))
for i in range(len(sentences)):
    X[i] = w2v_get_vec(sentences[i])
    if i % math.trunc(len(sentences) / 20) == 0:
        print(str(i) + "/" + str(len(sentences)))

# for sentence in sentences:
#     words = sentence.split()
#     for word in words:
#         if word in mydict:
#             j = mydict[word]
#             X[i][j] = X[i][j] + 1
#     i = i + 1

# pca = PCA(n_components=2)
# X_r = pca.fit(X).transform(X)
#
# lda = LinearDiscriminantAnalysis(n_components=2)
# X_r2 = lda.fit(X, y).transform(X)
#
# # Percentage of variance explained for each components
# print('explained variance ratio (first two components): %s'
#       % str(pca.explained_variance_ratio_))
#
# plt.figure()
colors = ['navy', 'turquoise', 'darkorange', 'brown', 'green', "black"]
# lw = 2
#
# for color, i, target_name in zip(colors, target_ids, target_names):
#     plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
#                 label=target_name)
# plt.legend(loc='best', shadow=False, scatterpoints=1)
# plt.title('PCA of messages dataset')
#
# plt.figure()
# for color, i, target_name in zip(colors, target_ids, target_names):
#     plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
#                 label=target_name)
# plt.legend(loc='best', shadow=False, scatterpoints=1)
# plt.title('LDA of messages dataset')
#
# plt.show()


def plot_LSA(test_data, test_labels, savepath="PCA_demo.csv", plot=True):
    lsa = TruncatedSVD(n_components=2)
    lsa.fit(test_data)
    lsa_scores = lsa.transform(test_data)
    color_mapper = {label: idx for idx, label in enumerate(set(test_labels))}
    color_column = [color_mapper[label] for label in test_labels]
    if plot:
        plt.scatter(lsa_scores[:, 0], lsa_scores[:, 1], s=4, alpha=.8, c=test_labels,
                    cmap=matplotlib.colors.ListedColormap(colors))
        patches = [mpatches.Patch(color=colors[i], label=target_names[i]) for i in range(len(colors))]
        plt.legend(handles=patches, prop={'size': 10})
    plt.show()

plot_LSA(X, y)