#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=suzhou
#SBTACH –gpus-per-task=1
#SBATCH --gres=gpu:1

source ~/.conda_init
conda activate deepchem
srun python run_experiment_LDS_on.py --dataset ${DATASET} --ue ${UE}