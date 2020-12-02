import pickle
from collections import Counter
from time import time
import matplotlib.pyplot as plt
import sys
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_roc_curve, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import SVC
from scipy.sparse import hstack
from sklearn.metrics import classification_report
from sklearn import preprocessing
from ScriptFeaturizer import scripts_to_tfidf
import warnings

def run_classification_models(features, budget_cats):
    # making test train splits
    X_train, X_test, y_train, y_test = train_test_split(features, budget_cats, test_size=0.2, random_state=0)

    # The Naive Bayes are bad, so I remove
    classifiers = [
        (LogisticRegression(random_state=0, max_iter = 1000), {
            'C': np.logspace(-2, 4, 10)
        }),
        (GradientBoostingClassifier(n_estimators=50, random_state=0), {
            'learning_rate': np.logspace(-4, 0, 5)
        }),
        (SVC(random_state=0), {
            'C': np.logspace(-2, 7, 5)
        })]

    for classifier, parameters in classifiers:
        print(classifier)

        clf = GridSearchCV(classifier, parameters, cv = 3)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        y_true, y_pred = y_test, clf.predict(X_test)

        print("Accuracy Score: \n")
        print(accuracy_score(y_test, y_pred))

        print("F1 Score: \n")
        print(f1_score(y_true, y_pred, average = 'macro'))
        print(classification_report(y_true, y_pred))

        disp = plot_roc_curve(clf, X_test, y_test)
        plt.show()

def ensemble_features(features1, features2):
    feats = []
    assert len(features1) == len(features2)
    for i in range(0,len(features1)):
        feats.append(np.append(features1[i], features2[i]))

    return np.array(feats)

def main():
    # setting up logging
    sys.stdout = open("results.txt", "w")
    warnings.filterwarnings('ignore')

    # reading in script metadata
    script_info = pd.read_csv('./data/IMSDB/final_movie_budgets.csv', sep=',')
    script_info['Budget'] = [int(bud.replace(',', '')) for bud in script_info['Budget']]  # reformatting budget

    # creating Budget Categories by quartile
    script_info['Bud_Cat'] = pd.qcut(script_info['Budget'], 2, labels=['low', 'high'])

    print(Counter(script_info['Bud_Cat']))

    # get list of scripts from data folder
    scripts = []
    for file in script_info['Filename']:
        with open(file, 'r') as txt:
            scripts.append(txt.read().replace('\n', ''))

    # create features from scripts
    scripts_bow, vocab = scripts_to_tfidf(scripts)
    scripts_ner = pickle.load(open('./data/features/ner_package_1084.pkl', 'rb'))[0]
    scripts_w2v, titles = pickle.load(open('./data/features/w2v_package_1084.pkl', 'rb'))
    scripts_w2v = preprocessing.scale(scripts_w2v)

    # run classifiers over data
    run_classification_models(scripts_bow, script_info['Bud_Cat'])
    run_classification_models(scripts_ner, script_info['Bud_Cat'])
    run_classification_models(scripts_w2v, script_info['Bud_Cat'])

    ### ENSEMBLING THE OTHER FEATURES ###
    # create ensemble of features

    bow_ner = hstack([scripts_bow, scripts_ner])
    bow_w2v = hstack([scripts_bow, scripts_w2v])
    wv2_ner = ensemble_features(scripts_w2v, scripts_ner)
    bow_ner_w2v = hstack([bow_ner, scripts_w2v])

    # try ensemble for all models
    run_classification_models(bow_ner, script_info['Bud_Cat'])
    run_classification_models(bow_w2v, script_info['Bud_Cat'])
    run_classification_models(wv2_ner, script_info['Bud_Cat'])
    run_classification_models(bow_ner_w2v, script_info['Bud_Cat'])

    # try ensemble for best model (SVM, C = 10)
    clf = SVC(random_state=0, C=10)
    X_train, X_test, y_train, y_test = train_test_split(bow_ner_w2v, script_info['Bud_Cat'], test_size=0.2, random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(clf)
    print("Accuracy Score: \n")
    print(accuracy_score(y_test, y_pred))
    print("F1 Score: \n")
    print(f1_score(y_test, y_pred, average='macro'))
    print(classification_report(y_test, y_pred))

    # create ensemble using prob_a
    clf = SVC(random_state=0, C=10, probability=True)
    X_train, X_test, y_train, y_test = train_test_split(scripts_bow, script_info['Bud_Cat'], test_size=0.2,
                                                        random_state=0)
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(scripts_bow)

    ### putting prob_a from BoW and other features together ###
    pbow_ner_w2v = ensemble_features(ensemble_features(probs, scripts_w2v), scripts_ner)

    clf = SVC(random_state=0, C=10)
    X_train, X_test, y_train, y_test = train_test_split(pbow_ner_w2v, script_info['Bud_Cat'], test_size=0.2,
                                                        random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(clf)
    print("Accuracy Score: \n")
    print(accuracy_score(y_test, y_pred))
    print("F1 Score: \n")
    print(f1_score(y_test, y_pred, average='macro'))
    print(classification_report(y_test, y_pred))

    ### DO PROB_A ENSEMBLE ABOVE BUT TEST on Logistic Regression instead ###
    # create ensemble using prob_a
    clf = LogisticRegression(random_state=0, C=10, max_iter = 1000, probability=True)
    X_train, X_test, y_train, y_test = train_test_split(scripts_bow, script_info['Bud_Cat'], test_size=0.2,
                                                        random_state=0)
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(scripts_bow)

    # putting prob_a from BoW and other features together
    pbow_ner_w2v = ensemble_features(ensemble_features(probs, scripts_w2v), scripts_ner)
    clf = LogisticRegression(random_state=0, max_iter = 1000, C=10)
    X_train, X_test, y_train, y_test = train_test_split(pbow_ner_w2v, script_info['Bud_Cat'], test_size=0.2,
                                                        random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(clf)
    print("Accuracy Score: \n")
    print(accuracy_score(y_test, y_pred))
    print("F1 Score: \n")
    print(f1_score(y_test, y_pred, average='macro'))
    print(classification_report(y_test, y_pred))


    # # visualizing features.
    # labelid = list(clf.classes_).index('high')
    # topn = sorted(zip(clf.coef_[labelid], vocab))[-10:]
    #
    # # lengths = [len(script) for script in scripts]
    # # plt.hist(lengths)

    # BERT with Sliding Window Results:
    # tp': 53, 'tn': 75, 'fp': 38, 'fn': 51,

    sys.stdout.close()

if __name__ == '__main__':
    main()
