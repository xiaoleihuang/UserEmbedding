import os
import subprocess
import time
import sys

dnames = ['yelp' , 'imdb', 'amazon_health'] # 'amazon', 'yelp' , 'imdb', 'amazon_health'
modes = {'skipgrams'} # 'skipgrams', 'word2user', 'doc2user', 'lda2user', 'bert2user'
cluster_nums = [4, 8, 12]

if not os.path.exists('./grid_scripts/'):
    os.mkdir('./grid_scripts/')

for dname in dnames:
    if dname == 'yelp':
        pattern_str = "qsub -l 'hostname=b*|c*,cpu=12,num_proc=12,mem_free=60g,ram_free=60g' -now no -cwd -o ./logs/ -e ./logs/ -q all.q {}"
    else:
        pattern_str = "qsub -l 'hostname=b*|c*,cpu=12,num_proc=12,mem_free=16g,ram_free=16g' -now no -cwd -o ./logs/ -e ./logs/ -q all.q {}"

    for mode in modes:
        for cluster_num in cluster_nums:
            if mode != 'skipgrams':
                sample_modes = ['']
            else:
                sample_modes = ['global', 'decay', 'local']

            for s_mode in sample_modes:
                script_file = './grid_scripts/evaluator_{}_{}_{}_{}.sh'.format(
                    dname, mode, cluster_num, s_mode
                )
                if os.path.exists(script_file):
                    os.remove(script_file)

                print('Running {}'.format(script_file))
                print()

                with open(script_file, 'w') as wfile:
                    wfile.write('source ~/.bashrc\n')
                    wfile.write('python evaluator.py {} {} {} {}\n'.format(
                        dname, mode, cluster_num, s_mode)
                    )
                command = pattern_str.format(script_file)
                time.sleep(1) # to avoid assign in the same machine
                process = subprocess.Popen(command, shell=True)
                print()

