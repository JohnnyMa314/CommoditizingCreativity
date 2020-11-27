import os
import re

import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

STOP_WORDS_HASH = set(STOP_WORDS)
nlp = spacy.load('en_core_web_sm')


def load_scripts(my_dir):
    scripts = sorted(os.listdir(my_dir))
    titles = [file.replace('.txt', '') for file in scripts]
    raw_scripts = []
    for file in scripts:
        file = open(os.path.join(my_dir, file), 'r')
        raw_scripts.append(file.read())
    return raw_scripts, titles


def lower(doc):
    return doc.lower()


def remove_stop_words(words):
    """Remove stop words. Using hashing for better performance."""
    return [word for word in words if word not in STOP_WORDS_HASH]


def tokenize_script(doc, stop_words=False):
    """Use regex to split text into list of words in text."""
    words = re.split("[^A-Za-z0-9\']+", lower(doc))
    if stop_words:
        words = remove_stop_words(words)
    return words


def clean_script(doc):
    """Clean out extraneous spacing. Useful for spacy parsers which can tokenize or lemmatize words."""
    return re.sub('\W+', ' ', lower(doc))


def make_doc_df(doc):
    """Run spacy model group document and return n rows where n corresponds to the number of total words in the document."""
    doc = clean_script(doc)
    tokens = nlp(doc)
    word_to_ner = {str(ent[0]): ent.label_ for ent in tokens.ents}
    words = []

    for token in tokens:
        words.append([token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
                      token.shape_, token.is_alpha, token.is_stop, word_to_ner.get(token.text)])

    return pd.DataFrame(words,
                        columns=['text', 'lemma', 'pos', 'tag', 'dep', 'shape', 'is_alpha', 'is_stop', 'ner_obj'])


def series_to_doc(ser):
    """Takes a pandas.core.series.Series and transforms to str containing the whole document"""
    return ' '.join(ser.values)
