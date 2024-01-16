#!/bin/bash
#
#SBATCH --job-name=__job_name__
#SBATCH --output=__out_path__.out
#SBATCH -e __out_path__.err
#SBATCH --partition=cpu
#
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=128G
#SBATCH --time=06:10:00         

export PYTHONPATH=$(pwd):$PYTHONPATH
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate specbm
__cmd_str__