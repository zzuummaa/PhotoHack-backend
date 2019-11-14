import json

trainingSet = json.load(open("trainingSet.json", encoding='utf-8', newline=''))

messages = list()
targets = list()
for example in trainingSet:
    if len(example["target"]) > 0:
        messages.append(example["text"])
        targets.append(example["target"])


trainingPairs = list(zip(messages, targets))
with open("trainingPairs.json", 'w', encoding='utf-8') as f:
    json.dump(trainingPairs, f, ensure_ascii=False)