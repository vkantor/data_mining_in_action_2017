import json
from collections import Counter


if __name__ == '__main__':
    res = Counter()
    def update_res(path):
        with open(path) as f:
            for key, value in json.load(f).iteritems():
                res[key] += value

    update_res('hw2/checkers/simple_gb_results.json')
    update_res('hw2/checkers/xgboost_params_results.json')
    update_res('hw3/checkers/text_classification_params_results.json')
    update_res('hw3/checkers/python_impl_svm_results.json')
    update_res('hw4/results.json')

    with open('all_nicknames', 'w') as f:
        for key, value in sorted(res.items(), key=lambda x: x[1], reverse=True):
            f.write(u'{} {}\n'.format(key, value).encode('utf-8'))
