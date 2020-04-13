import os
import subprocess
import time

flist = [item for item in os.listdir() if item.startswith('run_w') and item.endswith('.sh')]
pattern_str = "qsub -l 'hostname=b*|c*,cpu=16,num_proc=16,mem_free=16g,ram_free=16g' -now no -cwd -o ./logs/ -e ./logs/ -q all.q {}"

for item in flist:
    print('Running {}'.format(item))
    command = pattern_str.format(item)
    print(command)
    time.sleep(5)
    process = subprocess.Popen(command, shell=True)
