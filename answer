#!/usr/bin/python3

import sys
import itertools

from helper import *
import spacy
from enum import Enum
from spacy.symbols import *
from fuzzywuzzy import process

FUZZY_LIMIT = 3
FUZZY_CUTOFF = 50

nlp = spacy.load("en_core_web_sm") # need un-chunked data to identify question types
nlp_chunk = spacy.load("en_core_web_sm")

merge_ents = nlp.create_pipe("merge_entities")
merge_nps = nlp.create_pipe("merge_noun_chunks")

nlp.add_pipe(merge_ents)
nlp_chunk.add_pipe(merge_ents)
nlp_chunk.add_pipe(merge_nps)

class Qtype(Enum):
    WHO = "WHO"
    WHAT = "WHAT"
    WHEN = "WHERE"
    WHERE = "WHEN"
    WHY = "WHY"
    HOW = "HOW"
    YN = "YN"
    OTHER = "OTHER" # type not defined by current rules

def fuzzy_matches(sents, query):
    """
    Returns a list of strings containing at most FUZZY_LIMIT of the best matches to query 
    in doc. Matches are listed in decreasing order of best match

    sents is a list of strings to search
    query is the search string
    """
    results = process.extract(query, sents, limit=FUZZY_LIMIT)
    results.sort(reverse=True, key=lambda x: x[1])
    matches = [res[0] for res in results if query != None and res[1] > FUZZY_CUTOFF]
    return matches

def find_matches(doc, fixed_pre, q, fixed_post):
    """
    Creates a list of patterns (each a list of dictionary items) that can be passed 
    into a spacy Matcher

    fixed_pre a spacy object of the query sentence before the sentence position of q
    q a placeholder for the possible question answer
    fixed_pre a spacy object of the query sentence after the sentence position of q
    """
    sents = [s.text.strip() for s in doc.sents]
    pre_matches = fuzzy_matches(sents, fixed_pre.text) if fixed_pre != None else []
    post_matches = fuzzy_matches(sents, fixed_post.text) if fixed_post != None else []

    # get intersection of matching sentences, if both fixed_pre and fixed_post are not None
    post_set = set(post_matches)
    matches = [s for s in pre_matches if s in post_set]
    if not matches and not pre_matches:
        matches = post_matches
    elif not matches and not post_matches:
        matches = pre_matches
    return matches

def best_match(doc, fixed_pre, q, fixed_post):
    matches = find_matches(doc, fixed_pre, q, fixed_post)
    return matches[0] if matches else None

def process_be(doc, q):
    """
    Simply moves the copular 'be' to after the subject

    q is a spacy question sentence span
    Returns a new, independent spacy document object for the new sentence
    """
    for t in q.root.children:
        if t.dep == nsubj:
            subj = q[t.left_edge.i : t.right_edge.i + 1].text_with_ws.capitalize()
            pred = q[t.right_edge.i + 1].text_with_ws
            verb = q.root.text_with_ws.lower()
            s = nlp((subj + verb + pred)[:-1] + ".")
            match = best_match(doc, s, None, None)
            return match


def process_subj_aux(doc, q):
    """
    Simply moves the auxiliary verb to just after the subject.

    q is a spacy question sentence span
    Returns a new, independent spacy document object for the new sentence
    """
    for t in q.root.children:
        if t.dep == nsubj:
            subj = q[t.left_edge.i : t.right_edge.i + 1].text_with_ws.capitalize()
            pred = q[t.right_edge.i + 1].text_with_ws
            verb = q[0].text_with_ws.lower()
            s = nlp((subj + verb + pred)[:-1] + ".")
            match = best_match(doc, s, None, None)
            return match

def process_wh(doc, q):
    """
    """
    if q[0].dep == nsubj: # no wh-movement
        for t in q.root.children:
            if t.dep == attr:
                attribute = q[t.left_edge.i : t.right_edge.i + 1].text
                root = q.root.text
                s1 = nlp(" ".join([attribute, root]))
                s2 = nlp(" ".join([root, attribute]))
                match = best_match(doc, s1, None, None)
                best_match(doc, None, None, s2)
                # TODO: find a way to pick the better of the two
                return match
    else:
        q1 = q[1:]
        root = q.root.text_with_ws
        subj = ""
        rest = ""
        for t in q.root.children:
            if t.dep == nsubj:
                subj = q[t.left_edge.i : t.right_edge.i + 1].text
            else:
                rest = rest + q[t.left_edge.i : t.right_edge.i + 1].text
        s = nlp(" ".join([subj, root, rest]))
        match = best_match(doc, s, None, None)
        return match

def convert_to_query(doc, q):
    """
    Logic to generate a sentence with which to query the document for similarity
    For handled question types, syntactically converts the question into the form
    of a statement.
    """
    if q.root.lemma_ == "be":
        # yes/no question specifically with a form of be as a copular (linking) verb
        process_be(doc, q)
    if q[0].dep_ == "aux": 
        # yes/no question with subject-auxiliary inversion
        # Ignoring modal examples due to data
        return process_subj_aux(doc, q)
    # Default behavior is to simiply query the question
    return q

def get_qtype(q):
    """
    Returns a Qtype enum value corresponding to the type of question
    """
    if q[0].text.upper() == "WHO":
        return Qtype.WHO
    if q[0].text.upper() == "WHAT":
        return Qtype.WHAT
    if q[0].text.upper() == "WHEN":
        return Qtype.WHEN
    if q[0].text.upper() == "WHERE":
        return Qtype.WHERE
    if q[0].text.upper() == "WHY":
        return Qtype.WHY
    if q[0].text.upper() == "HOW":
        return Qtype.HOW
    if q.root.lemma_ == "be" or q[0].dep == aux:
        # copular shift or subject-auxiliary inversion
        return Qtype.YN
    return Qtype.OTHER

def answer_question(doc, q, qtype):
    """
    Calls appropriate rule-based functions to fetch answer for q from doc based on qtype

    Idea: create a query for doc consisting of the question, rearranged manually using
    English grammar rules, with the question words removed or replaced with generic token
    patterns as necessary. Then look for the best fuzzy match in doc.
    """
    if qtype == Qtype.WHO:
        return process_wh(doc, q)
    elif qtype == Qtype.WHAT:
        return process_wh(doc, q)
    elif qtype == Qtype.WHEN:
        return process_wh(doc, q)
    elif qtype == Qtype.WHERE:
        return process_wh(doc, q)
    elif qtype == Qtype.YN:
        if q.root.lemma_ == "be":
            return process_be(doc, q)
        elif q[0].dep == aux:
            return process_subj_aux(doc, q)
    else:
        return best_match(doc, q, None, None)

def get_answer(doc, question):
    qtype = get_qtype(question)
    if qtype in [Qtype.WHO, Qtype.WHAT, Qtype.WHEN, Qtype.WHERE, Qtype.YN]:
        a = answer_question(doc, question, qtype)
    else:
        # replace with whatever default non-rule-based method
        a = answer_question(doc, question, qtype)
    return a;

if __name__ == "__main__":
    article_path = sys.argv[1]

    article = read_article(article_path)
    qna, postags = parse_qna(1)

    with open(article_path) as f:
        text = f.read()
    doc = nlp(text)
    with open(sys.argv[2]) as f:
        qtext = f.readlines()

    for line in qtext:
        q = list(nlp(line.strip()).sents)[0]
        a = get_answer(doc, q)
        if a == None:
            most_similar_q, dataset_answer = get_most_similar_question(q.text, qna, postags)
            a = get_best_answer(most_similar_q, dataset_answer, article, postags)
        print(a.strip())
