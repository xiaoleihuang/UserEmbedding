import os
import subprocess
import time
import sys

dnames = ['yelp' , 'imdb', 'amazon_health'] # 'amazon', 'yelp' , 'imdb', 'amazon_health'
modes = {'skipgrams'} # 'skipgrams', 'word2user', 'doc2user', 'lda2user', 'bert2user'
cluster_nums = [4, 8, 12]
sample_modes = ['global', 'decay', 'local']
run_file = 'run_eval.sh'

with open(run_file, 'w') as wfile:
    wfile.write('#!/bin/sh\n')
#    wfile.write('source ~/.bashrc\n')

    for dname in dnames:
        for mode in modes:
            for cluster_num in cluster_nums:
                for s_mode in sample_modes:
                    wfile.write('python evaluator.py {} {} {} {}\n'.format(
                        dname, mode, cluster_num, s_mode)
                    )

process = subprocess.Popen('sh {} > results.txt'.format(run_file), shell=True)
time.sleep(3)
os.remove(run_file)
