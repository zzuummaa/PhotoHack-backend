from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow и tf.keras
import json

import numpy as np
import tensorflow as tf
from gensim import corpora
from tensorflow import keras

mydict = corpora.Dictionary.load('../one-hot_encoding.dict')
mydict = mydict.token2id
jsonData = json.load(open("../trainingPairs.json", encoding='utf-8', newline=''))
sentences = [i[0] for i in jsonData]
target = [i[1][0] for i in jsonData]

y = np.array(target)

X = np.zeros((len(sentences), len(mydict)))
i = 0
for sentence in sentences:
    words = sentence.split()
    for word in words:
        if word in mydict:
            j = mydict[word]
            X[i][j] = X[i][j] + 1
    i = i + 1

model = keras.Sequential([
    keras.layers.Dense(len(mydict), activation='relu'),
    keras.layers.Dense(11, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

target_names = np.asarray(["Отпуск", "Работа", "Проект, планировани", "Работа с компьютером", "Ура", "Нет"])

# train_labels = np.asarray(label)
model.fit(X, y, epochs=10)