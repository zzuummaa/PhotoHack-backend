#!flask/bin/python
import json

import numpy as np
from flask import Flask, request
from tensorflow_core.python.keras.models import load_model

from nlp_tools import prepare_sentence, extract_fitures, target_names, target_ids, template_ids

app = Flask(__name__)
model = load_model('my_model.h5')


class Prediction(object):
    def __init__(self, situation_id: int, situation_name: str, probability: float, **kwargs):
        self.situation_id = situation_id
        self.situation_name = situation_name
        self.probability = probability
        self.attribute = kwargs or None


class PredictionEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Prediction):
            return obj.__dict__
        return json.JSONEncoder.default(self, obj)


@app.route('/predict', methods=['POST'])
def predict():
    sentence = request.json['text']
    prepared = prepare_sentence(sentence)
    features = extract_fitures(prepared)
    featuresArr = np.asarray([features])
    predictions = model.predict(featuresArr)[0]

    out = list()
    i = 0
    while i < len(target_ids):
        id = target_ids[i]
        out.append(Prediction(template_ids[i], target_names[i], predictions[id].item()))
        i = i + 1
    return json.dumps(out, cls=PredictionEncoder, ensure_ascii=False)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8000)
