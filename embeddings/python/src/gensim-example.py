import pprint

document = "Human machine interface for lab abc computer applications"

text_corpus = [
    "Human machine interface for lab abc computer applications",
    "A survey of user opinion of computer system response time",
    "The EPS user interface management system",
    "System and human system engineering testing of EPS",
    "Relation of user perceived response time to error measurement",
    "The generation of random binary unordered trees",
    "The intersection graph of paths in trees",
    "Graph minors IV Widths of trees and well quasi ordering",
    "Graph minors A survey",
]

ignore_list = set('for a of the and to in'.split(' '))
# Lowercase each document, split it by white space and filter out ignored words
texts = [[word for word in document.lower().split() if word not in ignore_list]
         for document in text_corpus]

# Count word frequencies
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

# Only keep words that appear more than once
processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]
pprint.pprint(processed_corpus)

from gensim import corpora
dictionary = corpora.Dictionary(processed_corpus)
print(dictionary)
