from sklearn.model_selection import cross_val_score
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import numpy as np
import signal
import os
import json
import sys


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def signal_handler(signum, frame):
    raise Exception("Timed out!")


class Checker(object):
    def __init__(self):
        self.data = fetch_20newsgroups(
            subset='all', 
            categories=[
                'sci.electronics',
                'sci.space',
                'sci.med'
            ], 
            remove=('headers', 'footers', 'quotes')
        )

    def check(self, params_path):
        try:
            with open(params_path, 'r') as f:
                params = json.load(f)

            signal.signal(signal.SIGALRM, signal_handler)
            signal.alarm(60)
            pipeline = make_pipeline(
                CountVectorizer(**params['count_vectorizer_params']), 
                TfidfTransformer(**params['tfidf_transformer_params']), 
                LogisticRegression(**params['logistic_regression_params'])
            )
            score = np.mean(cross_val_score(
                pipeline, 
                self.data.data, 
                self.data.target,
                scoring='accuracy', 
                cv=3
            ))
        except:
            print "Unexpected error:", sys.exc_info()[0]
            score = None
        
        return score


if __name__ == '__main__':
    print Checker().check(SCRIPT_DIR + '/text_classification_params_example.json')
