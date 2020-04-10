#!/usr/bin/python3

import sys

import spacy
from spacy import displacy
from spacy.matcher import PhraseMatcher
from spacy.symbols import nsubj

nlp = spacy.load("en_core_web_lg")
matcher = PhraseMatcher(nlp.vocab)

def process_be(q):
    """
    Simply moves the copular 'be' to after the subject
    """
    for t in q.root.children:
        if t.dep == nsubj:
            subj = q[t.left_edge.i : t.right_edge.i + 1].text_with_ws.capitalize()
            pred = q[t.right_edge.i + 1].text_with_ws
            verb = q.root.text_with_ws.lower()
            return nlp((subj + verb + pred)[:-1] + ".")


def process_subj_aux(q):
    """
    Simply moves the auxiliary verb to just after the subject.
    """
    for t in q.root.children:
        if t.dep == nsubj:
            subj = q[t.left_edge.i : t.right_edge.i + 1].text_with_ws.capitalize()
            pred = q[t.right_edge.i + 1].text_with_ws
            verb = q[0].text_with_ws.lower()
            return nlp((subj + verb + pred)[:-1] + ".")

def convert_to_query(q):
    """
    Logic to generate a sentence with which to query the document for similarity
    For handled question types, syntactically converts the question into the form
    of a statement.
    """
    if q.root.lemma_ == "be":
        # yes/no question specifically with a form of be as a copular (linking) verb
        process_be(q)
    if q[0].dep_ == "aux": 
        # yes/no question with subject-auxiliary inversion
        # Ignoring modal examples due to data
        return process_subj_aux(question)
    # Default behavior is to simiply query the question
    return q



def get_answer(question):
    if False:
        return None
    else:
        return max(doc.sents, key=lambda x: convert_to_query(question).similarity(x))

if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        text = f.read()
    doc = nlp(text)
    with open(sys.argv[2]) as f:
        qtext = f.read()
    questions = nlp(qtext)

    answers = [get_answer(q) for q in questions.sents]

    for a in answers:
        print(a.text.strip())
