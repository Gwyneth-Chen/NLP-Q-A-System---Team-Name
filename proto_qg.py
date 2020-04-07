#!/usr/bin/python3

import sys
import spacy  # type: ignore
from spacy import displacy

nlp = spacy.load("en_core_web_sm")


if __name__ == "__main__":
    text = " ".join(sys.argv[1:])
    doc = nlp(text)
    for sentence in doc.sents:
        print(
            "For sentence "
            f'"{"".join(token.text_with_ws for token in sentence).strip()}":'
        )
        r = sentence.root
        main_subject = None
        for i, token in enumerate(r.children):
            if token.dep_ in ["nsubj", "nsubjpass"]:
                main_subject = token.idx

        question = (
            "".join(
                (token.text_with_ws if token.idx != main_subject else "what ")
                for token in sentence
            ).strip()[:-1]
            + "?"
        )

        if main_subject is not None:
            print("Found possible question:")
            print(f"  {question}")
