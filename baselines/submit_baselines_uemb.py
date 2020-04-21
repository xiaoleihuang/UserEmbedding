'''This is the script to submit running baselines to generate both user and product embeddings.
'''

import os
import subprocess
import time


# to save the job logs
if not os.path.exists('./logs/'):
    os.mkdir('./logs/')

tasks = ['imdb', 'yelp', 'amazon']
baselines = ['bert2vec'] # , 'doc2user', 'lda2user', 'word2user'
pattern_str = "qsub -l 'hostname=b*|c*,cpu=10,num_proc=10,mem_free=16g,ram_free=16g' -now no -cwd -o ./logs/ -e ./logs/ -q all.q {}"

for task in tasks:
    for baseline in baselines:
        script_file = 'run_{}_{}.sh'.format(baseline, task)
        print('Running {}'.format(script_file))
        with open(script_file, 'w') as wfile:
            wfile.write('NUM_CORES=15')
            wfile.write('export MKL_NUM_THREADS=$NUM_CORES OMP_NUM_THREADS=$NUM_CORES')
            wfile.write('source ~/.bashrc\n')
            wfile.write('python {}.py {}\n'.format(baseline, task))
        command = pattern_str.format(script_file)
        print(command)
        time.sleep(1)
        process = subprocess.Popen(command, shell=True)

time.sleep(1)

#for task in tasks:
#    for baseline in baselines:
#        script_file = 'run_{}_{}.sh'.format(baseline, task)
#        subprocess.Popen('rm ' + script_file, shell=True)
