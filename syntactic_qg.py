#!/usr/bin/python3

# TODO: question selection
# TODO: tense
# TODO: Possibly generate better questions by looking for more specific things?

import sys
import functools as ft
import rank
from typing import List

import spacy  # type: ignore

nlp = spacy.load("en_core_web_sm")
merge_nouns = nlp.create_pipe("merge_noun_chunks")
merge_ents = nlp.create_pipe("merge_entities")
nlp.add_pipe(merge_nouns)
nlp.add_pipe(merge_ents)

# import neuralcoref  # type: ignore

# neuralcoref.add_to_pipe(nlp)


def determine_wh_word(tok):
    if tok.ent_type_ in ["PERSON", "NORP", "ORG"]:
        return "who"
    if tok.tag_ in ["PRP", "PRP$"] and tok.text.lower() != "they":
        return "who"
    if tok.pos_ in ["PRON"] and tok.text.lower() != "they":
        return "who"
    if tok.ent_type_ in ["FAC", "LOC", "GPE"]:
        return "where"
    if tok.ent_type_ in ["TIME", "DATE"]:
        return "when"
    return "what"


def truecase(tok):
    tag = tok.tag_
    if tag not in ["NNP", "NNPS"]:
        text = tok.text
        return text[0].lower() + text[1:]
    else:
        return tok.text


# REQUIRES: tokens other than the first are truecased, and are text only
#
# truecasing can't be done here because we need information from the parse
# tree, but we can't manipulate that (say, to add the wh-word).
def construct_question(tokens):
    firstToken = tokens[0].capitalize()
    if not firstToken[0].isalpha():
        return []
    qtext = firstToken
    in_parenthetical = False
    for token in tokens[1:]:
        if token[0] in ["("]:
            in_parenthetical = True
            continue
        if in_parenthetical:
            if token[0] in [")"]:
                in_parenthetical = False
            continue
        if token[0].isalnum():
            qtext += " " + token
        else:
            qtext += token
    qtext = qtext.strip()
    if qtext[-1] not in ".!?":
        return []
    return [qtext[:-1] + "?"]


def make_yn_questions(s):
    is_negative = False
    aux = None
    subject_root = None

    for token in s.root.children:
        if token.dep_ in ["nsubj", "nsubjpass"]:
            subject_root = token
        if token.dep_ == "aux":
            aux = token
        if token.dep_ == "auxpass" and aux is None:
            aux = token
        if token.dep_ in ["neg"]:
            is_negative = True

    if aux is None or subject_root is None:
        return []

    tokens = []
    for token in s:
        if token == aux:
            continue
        if token == subject_root:
            tokens.append(aux)
            tokens.append(subject_root)
            continue
        tokens.append(token)

    qs = construct_question(list(map(truecase, tokens)))

    return [
        {"question": qtext, "answer": "No" if is_negative else "Yes", "qtype": "yesno",}
        for qtext in qs
    ]


def make_wh_questions(sentence):
    roots = []
    for token in sentence.root.children:
        if token.dep_ in ["nsubj", "nsubjpass"]:
            roots.append(token)
        if token.dep_ in ["dobj"]:
            roots.append(token)

    result = []
    for root in roots:
        whword = determine_wh_word(root)
        whindex = root.idx
        tokens = [truecase(t) if t != root else f"{whword}" for t in sentence]
        sent = construct_question(tokens)
        result += [
            {
                "question": qtext,
                "answer": root.text,
                "wh-word": whword,
                "wh-index": whindex,
                "qtype": "wh",
            }
            for qtext in sent
        ]

    return result


def convert_sentence(s):
    if s.root.pos_ not in ["VERB"]:
        return []
    return make_wh_questions(s) + make_yn_questions(s)


if __name__ == "__main__":
    rank.nlp = nlp
    with open(sys.argv[1]) as f:
        text = f.read()
    doc = nlp(text)
    candidates: List[str] = []
    for sentence in doc.sents:
        qs = convert_sentence(sentence)
        for q in qs:
            candidates.append(q)
    candidates.sort(key=rank.rank, reverse=True)
    if len(sys.argv) >= 3:
        amt: int = int(sys.argv[2])
        results = candidates[:amt]
        print("\n".join(q["question"] for q in candidates[:amt]))
    else:
        print("\n".join(q["question"] for q in candidates))
