#!/usr/bin/python

import arff
import csv, json

if __name__ == "__main__":
    import syntactic_qg

    nlp = syntactic_qg.nlp
nlp = None

features = []


def register(f):
    features.append(f)
    return f


@register
def length_feature(q):
    text = q["question"]
    doc = nlp(text)
    result = {"length": sum(1 for _ in doc)}

    def in_bucket(i):
        return 5 * i < sum(1 for _ in doc)

    result.update({f"length-bucket-{i}": int(in_bucket(i)) for i in range(9)})
    return result


def make_histograms(result, keys):
    def in_bucket(i, val):
        return i < val

    for key in keys:
        result.update(
            {f"{key}-bucket-{i}": int(in_bucket(i, result[key])) for i in range(9)}
        )

    return result


@register
def wh_question(q):
    return {"wh-word": int(q["qtype"] == "wh")}


@register
def grammar_features(q):
    text = q["question"]
    doc = nlp(text)

    entities = 0
    pronouns = 0
    numbers = 0
    clauses = 0

    root = next(doc.sents).root

    for token in doc:
        if token.ent_type_:
            entities += 1
        if token.pos_ in ["PRON"]:
            pronouns += 1
        if token.pos_ in ["NUM"]:
            numbers += 1
        if token.text == "," and token.idx < root.idx:
            clauses += 1

    result = {
        "numbers": numbers,
        "pronouns": pronouns,
        "entities": entities,
        "noun-phrases": len(list(doc.noun_chunks)),
        "preceding-clauses": clauses,
    }

    return make_histograms(result, list(result.keys()))


@register
def vagueness_features(q):
    text = q["question"]
    doc = nlp(text)

    count = 0
    for np in doc.noun_chunks:
        if np.text.startswith("the") or np.text.startswith("The"):
            count += not np[0].ent_type_

    for np in nlp(q["answer"]).noun_chunks:
        if np.text.startswith("the") or np.text.startswith("The"):
            count += not np[0].ent_type_

    result = {"vague_nps": count}
    return make_histograms(result, ["vague_nps"])


@register
def nominalized(q):
    return {"answer-nominalized": int(q["answer"].endswith("ing"))}


@register
def answer_in_question(q):
    return {"answer-in-question": int(q["answer"] in q["question"])}


def all_features(q):
    global features
    results = {k: v for d in map(lambda f: f(q), features) for k, v in d.items()}
    if "score" in q:
        results["score"] = q["score"]
    return results


def construct_arff(data):
    if not data:
        return
    training_set = [all_features(q) for q in data]
    canonical = training_set[0]
    formatted = {
        "description": "training data for 11-411",
        "relation": "nlpdata",
        "attributes": [
            (k, "INTEGER") if k != "score" else (k, "REAL") for k in canonical
        ],
        "data": [[feature_set[k] for k in feature_set] for feature_set in training_set],
    }
    with open("nlp_training_data.arff", "w") as f:
        arff.dump(formatted, f)


def rank(q):
    qfeats = all_features(q)
    return (
        -1.9538 * qfeats["length-bucket-5"]
        + -1.1507 * qfeats["wh-word"]
        + -2.7436 * qfeats["preceding-clauses"]
        + -2.0225 * qfeats["numbers-bucket-0"]
        + 3.0174 * qfeats["numbers-bucket-1"]
        + -6.6506 * qfeats["numbers-bucket-6"]
        + -1.9843 * qfeats["pronouns-bucket-1"]
        + 1.7176 * qfeats["entities-bucket-3"]
        + 2.8096 * qfeats["preceding-clauses-bucket-0"]
        + 2.9722 * qfeats["preceding-clauses-bucket-1"]
        + 6.2312 * qfeats["preceding-clauses-bucket-3"]
        + 9.5269
    )


def load_training_input(questions_file, outputs_file):
    with open(questions_file) as f:
        questions = json.load(f)
    scores = {}
    with open(outputs_file) as f:
        for index, score, *_ in csv.reader(f):
            index = int(index)
            if index in scores:
                total, count = scores[index]
                scores[index] = (total + int(score), count + 1)
            else:
                scores[index] = (int(score), 1)

    for index in scores:
        total, count = scores[index]
        questions[index]["score"] = total / count

    return [q for q in questions if "score" in q]
