import spacy
from nltk.tokenize import sent_tokenize, wordpunct_tokenize
from helper import *

article = read_article("data/set5/a3.txt")

nlp = spacy.load("en_core_web_sm")

questions = [
    "what is a cockroach?",
    "can you eat cockroaches?"
]

yesno = ["can", "are"]

for question in questions:
    nouns = []
    qdoc = nlp(question)

    for token in qdoc:
        if token.pos_ == "NOUN":
            nouns.append(token.text)
        
    print(nouns)

    