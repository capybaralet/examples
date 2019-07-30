#!/bin/bash
#SBATCH --account=rpp-bengioy        
#SBATCH --cpus-per-task=2              
#SBATCH --gres=gpu:1                   
#SBATCH --mem=4G                       
#SBATCH --time=3:00:00                 


# source the env
source activate base

# Copy the data locally
#rsync -avz $SCRATCH/data $SLURM_TMPDIR

# Copy the experiment script locally
#rsync -avz /home/capybara/examples/mnist/STML.py $SLURM_TMPDIR

# Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR
#exec python $SLURM_TMPDIR/STML.py --data_path=$SLURM_TMPDIR
exec python -u /home/capybara/examples/mnist/STML.py --verbosity=3 #--data_path=$SLURM_TMPDIR --verbosity=3

