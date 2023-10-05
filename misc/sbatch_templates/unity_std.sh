#!/bin/bash
#
#SBATCH --job-name=__job_name__
#SBATCH --output=__out_path__.out
#SBATCH -e __out_path__.err
#SBATCH --partition=longq
#
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=16G
#SBATCH --time=24:10:00         

export PYTHONPATH=$(pwd):$PYTHONPATH
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate specbm
__cmd_str__