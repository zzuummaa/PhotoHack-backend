import numpy as np
from tensorflow_core.python.keras.models import load_model
from nlp_tools import prepare_sentence, extract_fitures, target_names

model = load_model('my_model.h5')

sentence = "сижу на работе"
prepared = prepare_sentence(sentence)
features = extract_fitures(prepared)
featuresArr = np.asarray([features])
predictions = model.predict(featuresArr)

print(list(zip(target_names, predictions[0])))
