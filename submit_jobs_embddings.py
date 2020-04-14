import os
import subprocess
import time

tasks = ['imdb', 'yelp', 'amazon']
pattern_str = "qsub -l 'hostname=b*|c*,gpu=1,cpu=2,num_proc=2,mem_free=16g,ram_free=16g' -now no -cwd -o ./logs/ -e ./logs/ -q g.q {}"

for task in tasks:
    script_file = 'run_embedding_{}.sh'.format(task)
    print('Running {}'.format(script_file))
    with open(script_file, 'w') as wfile:
        wfile.write('source ~/.bashrc\n')
        wfile.write('python embeddings.py ' + task + '\n')
    command = pattern_str.format(script_file)
    print(command)
    time.sleep(1)
    process = subprocess.Popen(command, shell=True)
    
