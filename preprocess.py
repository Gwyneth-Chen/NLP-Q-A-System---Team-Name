import spacy
nlp = spacy.load("en_core_web_sm")

d = []

temp, outpath = "ADP ADV AUX PRON PUNCT:transformed5.txt".split(":")

i = 0

with open(f"qnadata/{outpath}", "w") as out:
    with open("qnadata/raw.txt") as f:
        for line in f:
            q,a = line.strip().split("\t")

            doc = nlp(q)

            transform = []

            for token in doc:
                if token.pos_ in temp.split(" "):
                    transform.append(token.text)

                else:
                    transform.append(token.pos_)

            transform = " ".join(transform)

            out.write(f"{q}\t{a}\t{transform}\n")

            i += 1
            print(f"   {i}", end ="\r")

