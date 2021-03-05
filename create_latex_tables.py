'''
Prints stats about dataset
'''
import argparse
import logging
import time
import glob
import os
import json
import pandas as pd
from pandas.core.algorithms import value_counts

ROLE_ORDER = ['agent', 'patient', 'theme', 'experiencer', 'recipient']
def get_data_stats(datadir, latex_dir):
    labels = {}
    for file in glob.glob(os.path.join(datadir, '*.csv')):
        tmp = file.split('/')[-1].split('.csv')[0].split('-')
        if len(tmp) == 2:
            role, split = tmp
            df = pd.read_csv(file)
            count_labels = df['LABEL'].value_counts()
            if role in labels:
                labels[role][split] = f'{count_labels[1]}'
            else:
                labels[role] = {}
                labels[role][split] = f'{count_labels[1]}'
        else:
            pass
    table = pd.DataFrame.from_records(labels)
    table = table[ROLE_ORDER]
    print(table)
    table = table.to_latex()
    with open(os.path.join(latex_dir, 'dataset_info.tex'), 'w') as fout:
        fout.write(table)

def get_results(results_dir, latex_dir):
    results = {}
    results['unbalanced'] = {}
    results['balanced'] = {}
    for role in ROLE_ORDER:
        file = os.path.join(results_dir, f'{role}/results.json')
        _j = json.load(open(file))
        results['unbalanced'][role] = _j['1']['f1-score']
    for role in ROLE_ORDER:
        file = os.path.join(results_dir, f'{role}-upsampled/results.json')
        _j = json.load(open(file))
        results['balanced'][role] = _j['1']['f1-score']

    # import ipdb;ipdb.set_trace()
    table = pd.DataFrame.from_records(results).T
    table = table[ROLE_ORDER]
    print(table)
    table = table.to_latex()
    with open(os.path.join(latex_dir, 'results.tex'), 'w') as fout:
        fout.write(table)

if __name__=="__main__":

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    start = time.time()
    get_data_stats('data', 'latex')
    get_results('models', 'latex')
    end = time.time()
    logging.info(f'Time to run script: {end-start} secs')