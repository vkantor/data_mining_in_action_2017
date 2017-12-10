import os
import json
import itertools


def calc_score(filename):
    hits_at_5 = 0.
    size = 0
    try:
        with open(filename) as f_pred, open('test_with_answers.txt') as f_ans:
            for line_pred, line_ans in itertools.izip(f_pred, f_ans):
                predictions = map(int, line_pred.split(','))
                answer = map(int, line_ans.split())[-1]
                size += 1
                hits_at_5 += answer in predictions[:5]
        return hits_at_5 / size
    except:
        return None


if __name__ == '__main__':
    scores = {}
    results = {}
    for filename in os.listdir('dmia_hw4'):
        if filename.endswith('.txt'):
            name = '_'.join(filename.split()[0].split('_')[-1:]).strip()
            #name = filename.split('-')[1].strip()[:-len('.txt')]
            score = calc_score('dmia_hw4/' + filename)
            print name, score

            if score is not None:
                results[name] = score
            else:
                scores[name] = 0.025

    best_score = max(results.values())
    for name in results:
        scores[name] = max(round(2 ** (15 * (results[name] - best_score)), 3), 0.05)

    with open('results.json', 'w') as f:
        json.dump(scores, f, indent=4)

    with open('results.csv', 'w') as f:
        for name in sorted(scores):
            f.write('{},{}\n'.format(name, scores[name]))
