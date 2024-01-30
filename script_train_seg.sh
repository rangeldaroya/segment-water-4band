#!/bin/bash  
#SBATCH -c 6  # Number of Cores per Task
#SBATCH --mem=16G  # Requested Memory
#SBATCH -p gypsum-1080ti  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -o slurm_trainseg/lraspp-%j.out  # %j = job ID
#SBATCH --job-name=seg-lraspp
#SBATCH --time=96:00:00 
#SBATCH --mail-type=ALL

python train_seg.py --model_type "lraspp" --input_type "4band" #--loss_type "adapmaxpool"