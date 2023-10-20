#!/bin/bash
#
#SBATCH --job-name=__job_name__
#SBATCH --output=__out_path__.out
#SBATCH -e __out_path__.err
#SBATCH --partition=cpu-long
#
#SBATCH -N 1
#SBATCH -n 64
#SBATCH --mem=512G
#SBATCH --time=48:10:00         

export PYTHONPATH=$(pwd):$PYTHONPATH
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate specbm
__cmd_str__