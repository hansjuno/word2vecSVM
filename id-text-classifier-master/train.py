import csv
import sys
import time
import dill
import pickle
import progressbar
import argparse
import numpy as np
import json
import pandas as pd
from pandas.io.json import json_normalize
from collections import defaultdict
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.wrappers import FastText
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, classification_report

# config
STOPWORDS_FILE = 'stopwords-id.txt'
RAW_DATASET_FILE = 'dataset_labeled.json'
WORD2VEC_C_FILE = '../model word embeddings/model_arif.bin'
MODEL_OUTPUT_FILE = 'model.pkl'


class TfidfEmbeddingVectorizer(BaseEstimator, TransformerMixin):

    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.values())

    def fit(self, X, y=None):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                # mean of each columns (axis=0) on 2d array
                # so each row will have same size
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])


class SimpleIndonesianPreprocessor(BaseEstimator, TransformerMixin):
    """
    Simple Indonesian text preprocessor
    """

    def __init__(self, stem=True, stopwords=True, verbose=True):
        self.stemmer = StemmerFactory().create_stemmer() if stem else None
        self.stopwords = []
        if stopwords:
            with open(STOPWORDS_FILE, 'r') as f:
                self.stopwords = f.read().splitlines()
        self.verbose = verbose

    def fit(self, X, y=None):
        return self

    def inverse_transform(self, X):
        return [" ".join(doc) for doc in X]

    def transform(self, X):
        results = []
        if self.verbose:
            print('Preprocessing..')
            bar = progressbar.ProgressBar()
            for doc in bar(X):            
                results.append(list(self.tokenize(doc)))
            return results
        else:
            return [
                list(self.tokenize(doc)) for doc in X
            ]

    def tokenize(self, document):
        if self.stemmer:      
            # stem and split by whitespaces    
            for token in self.stemmer.stem(document).split():
                if token not in self.stopwords:
                    yield token
        else:
            for token in document.lower().split():
                if token not in self.stopwords:
                    yield token
                    


## Methods


def identity(arg):
    """
    Simple identity function works as a passthrough.
    """
    return arg

def cross_validate_result_tabular_report(results, keys=None, decimals=4):
    """
    Convert cross_validate results dict/map to a tabular representation
    """
    if not keys:
        keys = results.keys()
    rownum = len(results[keys[0]])
    print('')
    print('\t'.join(['No'] + keys))
    total = {}
    # transpose results
    for i in range(rownum):
        row = [str(round(results[key][i], decimals)) for key in keys]
        print('\t'.join([str(i)] + row))                    
    print('')
    # average
    row_avg = [str(results[key].mean()) for key in keys]
    print('\t'.join(['Avg'] + row_avg))
    print('')

def build_and_evaluate(X_raw, y_raw, preprocessor, vectorizer, classifier, cv=None, outpath=None, verbose=True):

    # preprocessing
    X = preprocessor.fit_transform(X_raw)
    X = vectorizer.fit_transform(X)
    print(type(X))
    print(X.shape)
    
    # Label encode the targets
    labels = LabelEncoder()
    y = labels.fit_transform(y_raw)

    # Begin evaluation
    if verbose: 
        print("Cross-validating..")

    # cross validation    
    scorer = {
      'pos_recall': make_scorer(recall_score, pos_label=1),
      'neg_recall': make_scorer(recall_score, pos_label=0),
      'pos_precision': make_scorer(precision_score, pos_label=1),
      'neg_precision': make_scorer(precision_score, pos_label=0),
    }
    cv_results = cross_validate(classifier, X, y=y, scoring=scorer, cv=cv)
    
    # calculate f1
    cv_results['test_pos_f1'] = np.array([ 2 * cv_results['test_pos_recall'][i] * cv_results['test_pos_precision'][i] / (cv_results['test_pos_recall'][i] + cv_results['test_pos_precision'][i]) for i in range(len(cv_results['test_pos_recall'])) ])

    cv_results['test_neg_f1'] = np.array([ 2 * cv_results['test_neg_recall'][i] * cv_results['test_neg_precision'][i] / (cv_results['test_neg_recall'][i] + cv_results['test_neg_precision'][i]) for i in range(len(cv_results['test_neg_recall'])) ])

    if verbose:
        print('Classifier: {}'.format(type(classifier).__name__))
        cross_validate_result_tabular_report(cv_results, keys=['test_pos_f1', 'test_pos_precision', 'test_pos_recall', 'test_neg_f1', 'test_neg_precision', 'test_neg_recall'])

    # if outpath provided, dump trained model as pickle
    if outpath:

        if verbose:
            print("Building complete model and saving ...")

        # save complete pipeline
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('vectorizer', vectorizer),
            ('classifier', classifier),
        ])
            
        if verbose:
            # quick validation
            X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            y_predict = model.predict(X_test)
            print(classification_report(y_test, y_predict, target_names=labels.classes_))

        model.fit(X_raw, y_raw)
        model.labels_ = labels        

        if verbose:            
            with open(outpath, 'w') as f:
                pickle.dump(model, f)
            print("Model saved in {}".format(outpath))


if __name__ == '__main__':    

    models = {
        'tfidf-lsvm': 'tfidf-lsvm', 
        'w2v-rbfsvm': 'w2v-rbfsvm',
    }

    parser = argparse.ArgumentParser(description='Train and evaluate Indonesian text classifier')
    parser.add_argument('-m', '--model', default=models['w2v-rbfsvm'], choices=models.values(), help='Pipeline model')
    parser.add_argument('-o', '--output', help='path to save the trained model pickle file (model.pkl)')
    parser.add_argument('-s', '--silent', action='store_true', default=False, help='display no log in console')
    args = parser.parse_args()

    # load dataset
    X = []
    y = []  
    with open(RAW_DATASET_FILE, 'r') as f:
        dataset = json.load(f)
        df = pd.io.json.json_normalize(data=dataset)
        data_matrix = df.as_matrix()
        for line in data_matrix:
            y.append(line[0])
            X.append(line[1])
            

    verbose = not args.silent        

    # pipeline model 1
    if args.model == models['tfidf-lsvm']:
        preprocessor = SimpleIndonesianPreprocessor(verbose=verbose)
        vectorizer = TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False)
        classifier = SGDClassifier(max_iter=100, tol=None)    

    # pipeline model 2
    elif args.model == models['w2v-rbfsvm']:
        preprocessor = SimpleIndonesianPreprocessor(stem=False, verbose=verbose)
        # load w2v
        if verbose:
            print('Loading word2vec file {}..'.format(WORD2VEC_C_FILE))
        wv = KeyedVectors.load_word2vec_format(WORD2VEC_C_FILE, binary=False, unicode_errors="ignore")
        # wv = FastText.load_fasttext_format(WORD2VEC_C_FILE)
        if verbose:
            print('Indexing word2vec file..'.format(WORD2VEC_C_FILE))
        w2v = dict(zip(wv.index2word, wv.syn0))
        vectorizer = TfidfEmbeddingVectorizer(w2v)
        classifier = SVC()

    else:
        print('Unknown model:' + args.model)
        sys.exit()

    # build and evaluate model
    build_and_evaluate(X, y, preprocessor, vectorizer, classifier, cv=8, outpath=args.output, verbose=verbose)