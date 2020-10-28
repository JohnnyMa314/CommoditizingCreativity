import pandas as pd
import os
import numpy as np
import ast
import spacy
from time import time
from collections import Counter
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from ScriptFeaturizer import scripts_to_bow
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.extmath import density
from sklearn import metrics
nlp = spacy.load('en_core_web_sm')

def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."

def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

    print("classification report:")
    print(metrics.classification_report(y_test, pred, target_names=set(y_test)))

    print("confusion matrix:")
    print(metrics.confusion_matrix(y_test, pred))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time


def main():
    script_info = pd.read_csv(os.path.join(os.getcwd(), 'data/scraping/successful_files.csv'),
                              names = ['script', 'genre', 'stars', 'title', 'dir'],
                              sep = ';')

    scripts_bow, titles, vocab = scripts_to_bow(os.path.join(os.getcwd(), 'data/scraping/texts/')) # get sparse matrix of features
    script_info = script_info.drop_duplicates(['title']).sort_values(by=['title'])

    # check if lists are equal
    if Counter(list(script_info.title)) == Counter(titles):
        print("The lists are identical")

    # get 1st genre from list of list of genres
    multi_genre = [ast.literal_eval(genre) for genre in script_info.genre]
    genre = [genre[0] for genre in multi_genre]
    classes = set(genre)
    print(Counter(genre))

    # making test train splits
    X = scripts_bow
    y = genres

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Logistic Regression
    clf = LogisticRegression(random_state=0, )
    clf.fit(X_train, y_train)
    clf.predict(X[:2, :])

    benchmark(clf)

    # Naive Bayes
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    metrics.confusion_matrix(y_pred, y_test)

    benchmark(nb)

    ## Multi Class

    # binarize class identity
    y_multi = []
    for genre_set in multi_genre:
        bins = []
        # check if in each class
        for i, item in enumerate(classes):
            bins.append(int(item in multi_genre[0]))

        y_multi.append(bins)

    mlb = MultiLabelBinarizer()
    X_train, X_test, y_train, y_test = train_test_split(X, y_multi, test_size=0.33, random_state=42)

    clf = MultiOutputClassifier(KNeighborsClassifier()).fit(X_train, y_train)
    pred = clf.predict(X_test)
    clf.score(X_test, pred)

if __name__ == '__main__':
    main()