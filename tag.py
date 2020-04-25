import sys
import spacy

nlp = spacy.load("en_core_web_sm")
merge_nouns = nlp.create_pipe("merge_noun_chunks")
merge_ents = nlp.create_pipe("merge_entities")
nlp.add_pipe(merge_nouns)
nlp.add_pipe(merge_ents)

if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        text = f.readlines()

    for line in text:
        s = nlp(line)
        print(" ".join([str((t.text, t.dep_)) for t in s]))
