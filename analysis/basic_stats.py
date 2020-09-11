import numpy as np
import json
import subprocess

datasets = [
    'amazon_health', 'yelp', 'imdb'
]
root_dir = '../data/raw/'
stats = {}

for dname in datasets:
    print('Working on the ', dname)
    indir = root_dir + dname + '/'
    inpath = indir + dname + '.tsv'

    stats[dname] = {
        'num_doc': 0,
        'num_user': set(),
        'num_product': set(),
        'word_per_doc': [],
        'num_train': 0,
        'num_valid': 0,
        'num_test': 0,
    }
    # measure train, valid, test lengths
    stats[dname]['num_train'] = int(subprocess.check_output(
        ['wc', '-l', indir+'train.tsv']).decode('utf-8').split()[0])
    stats[dname]['num_valid'] = int(subprocess.check_output(
        ['wc', '-l', indir+'valid.tsv']).decode('utf-8').split()[0])
    stats[dname]['num_test'] = int(subprocess.check_output(
        ['wc', '-l', indir+'test.tsv']).decode('utf-8').split()[0])

    # load the dataset
    with open(inpath) as dfile:
        cols = dfile.readline()
        cols = cols.strip().split('\t')
        doc_idx = cols.index('text')
        user_idx = cols.index('uid')
        product_idx = cols.index('bid')

        for line in dfile:
            line = line.strip()
            if len(line) < 5:
                continue

            line = line.split('\t')
            stats[dname]['num_doc'] += 1
            stats[dname]['num_user'].add(line[user_idx])
            stats[dname]['num_product'].add(line[product_idx])
            stats[dname]['word_per_doc'].append(len(line[doc_idx].split()))

    stats[dname]['num_user'] = len(stats[dname]['num_user'])
    stats[dname]['num_product'] = len(stats[dname]['num_product'])
    stats[dname]['mean_word_per_doc'] = np.mean(stats[dname]['word_per_doc'])
    stats[dname]['median_word_per_doc'] = np.median(stats[dname]['word_per_doc'])
    stats[dname]['75_percent_word_per_doc'] = np.percentile(stats[dname]['word_per_doc'], 75)
    del stats[dname]['word_per_doc']

with open('stats.json', 'w') as wfile:
    wfile.write(json.dumps(stats))
