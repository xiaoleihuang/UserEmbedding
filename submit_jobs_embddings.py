import os
import subprocess
import time
import sys

mode = sys.argv[1]
tasks = ['amazon_health'] # 'imdb', 'yelp', 'amazon', 'amazon_health'
mnames = ['bert'] # 'lda', 'word2vec', 'doc2vec', 'bert'

if mode == 'cpu':
    pattern_str = "qsub -l 'hostname=b*|c*,cpu=12,num_proc=12,mem_free=16g,ram_free=16g' -now no -cwd -o ./logs/ -e ./logs/ -q all.q {}"
else:
    pattern_str = "qsub -l 'hostname=b1[12345678]*|c*,gpu=1,cpu=2,num_proc=2,mem_free=16g,ram_free=16g' -now no -cwd -o ./logs/ -e ./logs/ -q g.q {}"

# remove the previous generated script
for task in tasks:
    for mname in mnames:
        script_file = 'run_embedding_{}_{}.sh'.format(task, mname)
        if os.path.exists(script_file):
            os.remove(script_file)

for task in tasks:
    for mname in mnames:
        script_file = 'run_embedding_{}_{}.sh'.format(task, mname)
        print('Running {}'.format(script_file))
        print()
        with open(script_file, 'w') as wfile:
            wfile.write('source ~/.bashrc\n')
            wfile.write('python embeddings.py {} {}\n'.format(task, mname))
        command = pattern_str.format(script_file)
        print(command)
        print()
        time.sleep(3) # to avoid assign in the same machine
        process = subprocess.Popen(command, shell=True)

