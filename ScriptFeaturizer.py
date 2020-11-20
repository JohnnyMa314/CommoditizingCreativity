import os
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from spacy.lang.en.stop_words import STOP_WORDS
nlp = spacy.load('en_core_web_sm')

'''
INPUT: a directory containing a .txt file of scripts
OUTPUT: a vector of textual features
'''

def load_scripts(my_dir):
    scripts = sorted(os.listdir(my_dir))
    titles = [file.replace('.txt', '') for file in scripts]
    raw_scripts = []
    for file in scripts:
        file = open(os.path.join(my_dir, file), 'r')
        raw_scripts.append(file.read())
    return raw_scripts, titles

# BAG OF WORDS MODEL
def scripts_to_bow(scripts):
    
    # custom stop words for scripts
    film_stop_words = ['V.O.', "Scene", "CUT TO", "FADE IN"]
    stop_words = STOP_WORDS.union(film_stop_words)

    # vectorize scripts into BoW
    vectorizer = TfidfVectorizer(input='content', stop_words=stop_words, min_df = 0.2, ngram_range=(1,1)) # less than 0.2 frequency. bad.
    bow = vectorizer.fit_transform(scripts)
    vocab = vectorizer.get_feature_names()
    dict = vectorizer.vocabulary_

    return bow, vocab

# WORD EMBEDDINGS MODEL
def script_to_embeddings(directory, filename):
    features = []
    # word2vec in here
    return features

# TRANSFORMERS MODEL
def script_to_transformer(directory, filename):
    features = []
    # pretrained BERT in here
    return features