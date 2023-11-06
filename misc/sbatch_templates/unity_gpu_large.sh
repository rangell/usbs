#!/bin/bash
#
#SBATCH --job-name=__job_name__
#SBATCH --output=__out_path__.out
#SBATCH -e __out_path__.err
#SBATCH --partition=gypsum-1080ti
#
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -G 1
#SBATCH --mem=32G
#SBATCH --time=48:10:00         

export PYTHONPATH=$(pwd):$PYTHONPATH
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate specbm-gpu
__cmd_str__