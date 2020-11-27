from time import time

import pandas as pd
import spacy
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.extmath import density
from ScriptFeaturizer import scripts_to_tfidf

nlp = spacy.load('en_core_web_sm')


def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."


# to quickly produce nice benchmark metrics
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
    # reading in script metadata
    script_info = pd.read_csv('./data/IMSDB/final_movie_info.csv', sep=',')
    script_info = script_info[~script_info['Budget'].isna()]  # removing NA budgets
    script_info['Budget'] = [int(bud.replace(',', '')) for bud in script_info['Budget']]  # reformatting budget

    # creating Budget Categories
    script_info['Bud_Cat'] = pd.qcut(script_info['Budget'], 2, labels=["low", 'high'])

    # plt.hist(script_info['Budget'], bins = 3)
    # plt.show()

    # get list of scripts from data folder
    scripts = []
    for file in script_info['Filename']:
        with open(file, 'r') as txt:
            scripts.append(txt.read().replace('\n', ''))

    # create features from scripts
    scripts_bow, vocab = scripts_to_tfidf(scripts)

    # making test train splits
    X = scripts_bow
    y = script_info['Bud_Cat']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=420)

    # Logistic Regression
    clf = LogisticRegression(random_state=420)
    benchmark(clf)

    # Naive Bayes
    nb = MultinomialNB()
    benchmark(nb)

    # visualizing features.
    labelid = list(clf.classes_).index('high')
    topn = sorted(zip(clf.coef_[labelid], vocab))[-10:]


if __name__ == '__main__':
    main()
