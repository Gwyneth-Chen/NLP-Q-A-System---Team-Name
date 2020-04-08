#!/usr/bin/python3

import sys
import functools as ft

import spacy  # type: ignore
from spacy import displacy

nlp = spacy.load("en_core_web_sm")


def collect_indices(t):
    return ft.reduce(set.union, (collect_indices(c) for c in t.children), {t.idx})


def make_wh_question(blank_root, sentence):
    if blank_root is None:
        return None

    indices = collect_indices(blank_root)
    return (
        "".join(
            (
                token.text_with_ws
                if token.idx not in indices
                else "what "
                if token.idx <= min(indices)
                else ""
            )
            for token in sentence
        ).strip()[:-1]
        + "?"
    )


def convert_sentence(s):
    subject_root = None
    obj_root = None

    for token in s.root.children:
        if token.dep_ in ["nsubj", "nsubjpass"]:
            subject_root = token
        if token.dep_ in ["pobj", "dobj"]:
            obj_root = token
        if token.dep_ in ["prep"]:
            for t in token.children:
                if t.dep_ in ["pobj", "dobj"]:
                    obj_root = token
    return make_wh_question(subject_root, s), make_wh_question(obj_root, s)


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        text = f.read()
    doc = nlp(text)
    for sentence in doc.sents:
        question1, question2 = convert_sentence(sentence)

        print(
            "For sentence "
            f'"{"".join(token.text_with_ws for token in sentence).strip()}":'
        )

        print("Found possible questions:")
        print(f"  {question1, question2}")
