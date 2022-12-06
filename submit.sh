#!/bin/bash
#
#SBATCH --job-name=test
#
#SBATCH --time=1440:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G

ml python/3.6.1
pip install --user -r requirements.txt
srun python3 main.py