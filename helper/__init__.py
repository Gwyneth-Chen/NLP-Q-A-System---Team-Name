from helper.cossim import *
import spacy

from nltk.tokenize import sent_tokenize, wordpunct_tokenize

nlp = spacy.load("en_core_web_sm")

def read_article(filepath):
    """
        reads article selected in filepath
        returns a list of sentences (ntlk sent_tokenized)
    """
    out = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                out.extend(sent_tokenize(line))
    return out

def parse_qna(n):
    """
        parse qna dataset selected in filepath
        n: dataset number: the higher the number, the fewer pos tags used
    """

    def read_meta(n):
        """
            read metadata from qna/meta.txt 
            return data telling us which qna dataset to use
        """
        meta = [""]
        with open("qnadata/meta.txt") as f:
            for line in f:
                postags, filepath = line.strip().split("=")
                meta.append((postags, filepath))
        
        return meta[n]

    postags, filepath = read_meta(n)

    out = []
    with open(f"qnadata/{filepath}") as f:
        for line in f:
            q,a,pos = line.strip().split("\t")
            pos = tuple(pos.split(" "))

            out.append((q,a,pos))

    return out, postags


def get_spacy_tags(line, postags):
    out = []
    postags = set(postags.split(" "))
    for token in nlp(line):
        if token in postags: out.append(token.text)
        else: out.append(token.pos_)
    return out

def get_most_similar_question(question, qna, postags):
    """
        find the question in the qna dataset that most resembles the input question

        1. pos tag cosine similarity
        2. raw question similarity
    """

    question_split = wordpunct_tokenize(question)
    question_pos = get_spacy_tags(question, postags)

    best_question, dataset_answer, best_score = None, None, 0

    for q,a,pos in qna:
        score = cosine_similarity_unigram(question_split, wordpunct_tokenize(q))
        score += cosine_similarity_bigram(question_pos, pos)

        if score > best_score:
            best_score = score
            best_question = q
            dataset_answer = a

    return best_question, dataset_answer

def get_best_answer(most_similar_question, dataset_answer, article, postags, limit=5):
    """
        dataset_answer: string
            - answer of the best question in qna dataset
        
        article: list of sentences (str)
    """

    question_split = wordpunct_tokenize(most_similar_question)
    question_tags = get_spacy_tags(most_similar_question, postags)

    dataset_answer_split = wordpunct_tokenize(dataset_answer)
    dataset_answer_tags = get_spacy_tags(dataset_answer, postags)
    best_answer, best_score = None, 0

    for line in article:

        if len(wordpunct_tokenize(line)) < limit:
            continue

        line_split = wordpunct_tokenize(line)
        line_tags = get_spacy_tags(line, postags)

        score = cosine_similarity_bigram(line_tags, dataset_answer_tags)
        score += 2*cosine_similarity_unigram(question_split, line_split)
        score /= len(line_split)

        if score > best_score:
            best_score = score
            best_answer = line
    
    return best_answer





