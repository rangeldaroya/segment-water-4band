#!/bin/bash  
#SBATCH -c 6  # Number of Cores per Task
#SBATCH --mem=48G  # Requested Memory
#SBATCH -p gypsum-2080ti  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -o slurm/whole-%j.out  # %j = job ID
#SBATCH --job-name=lang-sam-whole
#SBATCH --time=01:00:00 
#SBATCH --mail-type=ALL

python sample_whole.py --i 0