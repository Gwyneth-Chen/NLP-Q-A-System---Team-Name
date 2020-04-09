#!/usr/bin/python3

import sys
import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_lg")

def get_answer(question):
    return max(doc.sents, key=lambda x: question.similarity(x))

if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        text = f.read()
    doc = nlp(text)
    with open(sys.argv[2]) as f:
        qtext = f.readlines()

    answers = [get_answer(nlp(line.strip())) for line in qtext]

    for a in answers:
        print(a.strip())
