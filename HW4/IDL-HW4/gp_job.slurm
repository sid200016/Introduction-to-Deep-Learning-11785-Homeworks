#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH -t 08:00:00
#SBATCH -A cis250019p
#SBATCH -J my_gpu_job
#SBATCH -o my_gpu_job.out

# Load necessary modules
module load cuda/12.2
module load anaconda3

# (Optional) Activate your conda environment
# source activate myenv
conda activate /ocean/projects/cis250019p/mzhang23/TA/HW4/envs/hw4_env && export PYTHONNOUSERSITE=1
# Run your code
python hw4p2.py