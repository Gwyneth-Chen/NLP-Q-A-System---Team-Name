import os
import sys
from nltk.tokenize import wordpunct_tokenize

class Parser():
    def __init__(self):
        self.parse()

    def parse(self):
        self.set_corpus()
        self.set_word_frequency()
        self.replace_unknowns()
        self.ngram = {}
        self.set_ngram(n=2)
        self.set_ngram(n=3)

    def set_corpus(self):
        """
            create self.corpus:
                list containing all sentences 
        """
        self.corpus = []

        for root, subdirs, files in os.walk("Development_data"):
            for filename in files:
                if ".txt" in filename:
                    filepath = root + "/" + filename

                    with open(filepath) as f:
                        for line in f:
                            line = line.strip()
                            if len(line) > 0:
                                self.corpus.append(wordpunct_tokenize(line))


    def set_word_frequency(self):
        """
            create self.freq: dictionary containing frequencies of all words
        """
        self.freq = {}
        for line in self.corpus:
            for word in line:
                if word not in self.freq:
                    self.freq[word] = 1
                else:
                    self.freq[word] += 1


    def replace_unknowns(self, threshold=5):
        """
            replace words with frequencies less than threshold with "<UNKNOWN>"
        """
        
        for line in self.corpus:
            for i,word in enumerate(line):
                if self.freq[word] < threshold:
                    line[i] = "<UNKNOWN>"

    def set_ngram(self, n=2):
        """
            create self.ngram: dictionary
                - key == number of grams, value == ngram dictionary
                    eg. self.ngram[2] will return the bigram dictionary
                    eg. self.ngram[3] will return the trigram dictionary
        """
        d = {}

        for line in self.corpus:
            if len(line) < n:
                continue
        
            for i in range(len(line)-n+1):
                ngram = tuple(line[i:i+n])

                if ngram not in d:
                    d[ngram] = 1
                else:
                    d[ngram] += 1
            
        self.ngram[n] = d


if __name__ == "__main__":
    p = Parser()
