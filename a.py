from helper import *

article_path = "data/set5/a3.txt"
article = read_article(article_path)
qna, postags = parse_qna(1)

questions = [
    "what is a cockroach?",
    "where do cockroaches live?",
    "can i be a cockroach?",
    "am i allowed to keep cockroaches as pets?",
    "who lives in a pineapple under the sea?"
]

for question in questions:
    """
        1. find most similar question from qna dataset

    """

    most_similar_question, dataset_answer = get_most_similar_question(question, qna, postags)
    best_answer = get_best_answer(most_similar_question, dataset_answer, article, postags)

    print(question, best_answer)