import os
import time
import argparse

parser = argparse.ArgumentParser()
# ENVIRONMENT SETTINGS:
parser.add_argument('--save_dir', type=str, default=os.environ['SCRATCH'])
parser.add_argument('--bs', type=int, default=100)
#
args = parser.parse_args()
print (args)
args_dict = args.__dict__
args_str = " --".join([""] + [k + "=" + str(v) for k,v in args_dict.items()]) + " " 
print (args_str)
locals().update(args_dict)


bash_script_name= os.path.join(os.environ['HOME'], 'examples/mnist', 'niagara_launch_helper.sh ')
python_script_name= os.path.join(os.environ['HOME'], 'examples/mnist', 'STML.py ')

def wwrite(f_, ss):
    f_.write(ss + '\n')

with open('niagara_launcher.sh', 'a') as f_:  
    f_.write(
    '#!/bin/bash')
    f_.write('\n')
    f_.write(
    '#SBATCH --nodes=1')
    f_.write('\n')
    f_.write(
    '#SBATCH --ntasks-per-node=40')
    f_.write('\n')
    f_.write(
    '#SBATCH --time=48:00:00')
    f_.write('\n')
    f_.write(
    '#SBATCH --job-name serialx40')
    f_.write('\n')
    f_.write(
    '#SBATCH --account=def-bengioy')
    f_.write('\n')
    f_.write(
    'source activate base')
    f_.write('\n')
    f_.write(
    'export OMP_NUM_THREADS=1')

    # START HERE
    args_str = ' --batch-size=' + str(bs)
    for starting_seed in range(5):
        for shuffle in [True, False]:
            for thresh in [-np.inf, -.001, 0, .001]:#,10]:
                ss = ' --starting_seed=' + str(starting_seed) + ' --environment_swapping=' + str(environment_swapping) + ' --PBT_interval=' + str(PBT_interval) 
                launch_str = " ".join([bash_script_name, python_script_name, args_str, ss])
                f_.write('\n')
                f_.write('(' + launch_str + ' && echo "job 01 finished") &')
    f_.write('\n')
    f_.write('wait')

os.system('chmod 755 niagara_launcher.sh')
#os.system('./niagara_launcher.sh')









