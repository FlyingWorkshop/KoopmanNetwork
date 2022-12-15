#!/bin/bash
#
#SBATCH --job-name=test
#
#SBATCH --time=10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

ml python/3.6.1
pip install --user -r requirements.txt
srun python3 main.py
