from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from tensorflow import keras

import nlp_tools

# TensorFlow и tf.keras

X, y = nlp_tools.load_training_set()

model = keras.Sequential([
    keras.layers.Dense(len(nlp_tools.mydict.token2id), activation='relu'),
    keras.layers.Dense(11, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

target_names = np.asarray(["Отпуск", "Работа", "Проект, планировани", "Работа с компьютером", "Ура", "Нет"])

# train_labels = np.asarray(label)
model.fit(X, y, epochs=20)
model.save('my_model.h5')
print("Model saved to my_model.h5")
