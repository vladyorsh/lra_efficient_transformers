#!/bin/bash

#SBATCH -J cls_luna
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err
#SBATCH -p gpu-troja
##SBATCH -q high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --constraint="gpuram40G|gpuram48G"
#SBATCH -D /lnet/troja/work/people/yorsh/lra

# Print each command to STDERR before executing (expanded), prefixed by "+ "
set -o xtrace
set -e

echo "Starting job at $(pwd)"
echo "Visible devices: $CUDA_VISIBLE_DEVICES"

module load transformers_module
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'
python3 example_cls.py
