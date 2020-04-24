#!/usr/bin/python

import syntactic_qg as qg
import json
import random

nlp = qg.nlp

data = []

for s in [1, 2, 3, 4, 5]:
    i = random.choice(range(1, 11))
    fname = f"Development_data/set{s}/a{i}.txt"
    print(f"generating questions for {fname}")
    with open(fname) as f:
        text = f.read()

    doc = nlp(text)
    for sentence in doc.sents:
        data += qg.convert_sentence(sentence)

print(len(data))
with open("training_qs_unrated.json", "w") as f:
    json.dump(data, f)
