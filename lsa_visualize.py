import json

import matplotlib
import matplotlib.patches as mpatches
from sklearn.decomposition import TruncatedSVD
from nlp_tools import target_names
import matplotlib.pyplot as plt
import numpy as np

colors = ['navy', 'red', 'turquoise', 'darkorange', 'brown', 'green', "black"]

data = json.load(open("w2vVectors.json", encoding='utf-8', newline=''))
data = list(zip(*data))
sentences = data[0]
X = np.asarray(data[1])
y = np.asarray(data[2])

def plot_LSA(test_data, test_labels, plot=True):
    lsa = TruncatedSVD(n_components=2)
    lsa.fit(test_data)
    lsa_scores = lsa.transform(test_data)
    if plot:
        plt.figure(dpi=300)
        plt.scatter(lsa_scores[:, 0], lsa_scores[:, 1], s=1, alpha=.8, c=test_labels,
                    cmap=matplotlib.colors.ListedColormap(colors))
        patches = [mpatches.Patch(color=colors[i], label=target_names[i]) for i in range(len(colors))]
        plt.legend(handles=patches, prop={'size': 6})
    plt.show()
    return lsa_scores


lsa_scores = plot_LSA(X, y)
