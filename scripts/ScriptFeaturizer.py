import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en.stop_words import STOP_WORDS

from FeatureUtils import tokenize_script


def num_lines(script):
    """Run regex to determine the number of speaking lines in a script."""
    return len(re.findall('[A-Z]{3,}[A-Za-z0-9 ]{,20}\\n\ {3,}', script))


def scripts_to_tfidf(scripts):
    """Create Tfidf matrix from tokenized scripts."""
    # custom stop words for scripts
    film_stop_words = ['V.O.', "Scene", "CUT TO", "FADE IN"]
    stop_words = STOP_WORDS.union(film_stop_words)

    # vectorize scripts into Tfidf matrix
    vectorizer = TfidfVectorizer(input='content', stop_words=stop_words, min_df=0.2,
                                 ngram_range=(1, 2))  # less than 20% frequency words are removed.
    bow = vectorizer.fit_transform(scripts)
    vocab = vectorizer.get_feature_names()

    return bow, vocab


def df_to_stats(df, groupby_col, stat='count'):
    """Computes a df containing a row for each values to a summary statistic."""
    if stat == 'count':
        summary_df = df.groupby(by=groupby_col).count().iloc[:, 0]
    return summary_df.values, summary_df.index.values


def scripts_to_embeddings(scripts):
    """Create WordEmbedding from tokenized scripts."""
    docs = [tokenize_script(script, stop_words=True) for script in scripts]
    
    model = Word2Vec(docs, 
                     min_count=1, 
                     size=300, 
                     window=5
                    )

    return model


def script_to_transformer(scripts):
    features = []
    # pretrained BERT in here
    return features
