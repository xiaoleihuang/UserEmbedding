import os
import subprocess
import time
import sys

tasks = ['word', 'word_user', 'word_user_product']
dnames = ['amazon_health', 'imdb', 'yelp']
modes = ['local', 'global', 'decay']
pattern_str = "qsub -l 'hostname=b*|c*,cpu=12,num_proc=12,mem_free=16g,ram_free=16g' -now no -cwd -o ./logs/ -e ./logs/ -q all.q {}"
grid_dir = './grid_scripts/'
if not os.path.exists(grid_dir):
    os.mkdir(grid_dir)

for task in tasks:
    for dname in dnames:
        if task == 'word':
            script_file = '{0}_{1}.sh'.format(task, dname)
            print('Running {}'.format(script_file))
            print()

            with open(grid_dir + script_file, 'w') as wfile:
                wfile.write('#!/bin/bash\n\n')
                wfile.write('source ~/.bashrc\n')
                wfile.write('cd /export/b10/xhuang/xiaolei_data/UserEmbedding/\n')
                wfile.write('python {0}.py {1}\n'.format(task, dname))
            command = pattern_str.format(script_file)
            process = subprocess.Popen(command, shell=True)
        else:
            for mode in modes:
                script_file = '{0}_{1}_{2}.sh'.format(task, dname, mname)
                print('Running {}'.format(script_file))
                print()

                with open(grid_dir + script_file, 'w') as wfile:
                    wfile.write('#!/bin/bash\n\n')
                    wfile.write('source ~/.bashrc\n')
                    wfile.write('cd /export/b10/xhuang/xiaolei_data/UserEmbedding/\n')
                    wfile.write('python {0}.py {1} {2}\n'.format(task, dname, mname))
                command = pattern_str.format(script_file)
                process = subprocess.Popen(command, shell=True)

