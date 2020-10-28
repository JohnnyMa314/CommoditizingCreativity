import os
from os import listdir
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from spacy.tokenizer import Tokenizer
from spacy.lang.en.stop_words import STOP_WORDS
nlp = spacy.load('en_core_web_sm')

'''
INPUT: a directory containing a .txt file of scripts
OUTPUT: a vector of textual features
'''
def scripts_to_bow(dir):
    scripts = sorted(os.listdir(dir))
    titles = [file.replace('.txt', '') for file in scripts]
    raw_scripts = []
    for file in scripts:
        file = open(os.path.join(dir, file), 'r')
        raw_scripts.append(file.read())

    # custom stop words for scripts
    film_stop_words = ['V.O.', "Scene", "CUT TO", "FADE IN"]
    stop_words = STOP_WORDS.union(film_stop_words)

    # vectorize scripts into BoW
    vectorizer = TfidfVectorizer(input='content', stop_words=stop_words, min_df = 0.2, ngram_range=(1,1)) # less than 0.2 frequency.
    bow = vectorizer.fit_transform(raw_scripts)
    vocab = vectorizer.get_feature_names()
    dict = vectorizer.vocabulary_

    return bow, titles, vocab

def script_to_embeddings(directory, filename):
    features = []
    return features

def script_to_transformer(directory, filename):
    features = []
    return features