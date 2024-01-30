#!/bin/bash  
#SBATCH -c 6  # Number of Cores per Task
#SBATCH --mem=48G  # Requested Memory
#SBATCH -p gypsum-2080ti  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -o slurm/%j-%a.out  # %j = job ID
#SBATCH --job-name=lang-sam
#SBATCH --time=01:00:00 
#SBATCH --mail-type=ALL
#SBATCH --array=0-31%10   # 10 jobs at a time

variations=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31)
casenumidx=$((SLURM_ARRAY_TASK_ID))
echo "Running i=${variations[$casenumidx]}"

python sample.py --i ${variations[$casenumidx]} --text_prompt "river" --thresh 0.35 --num_split 10