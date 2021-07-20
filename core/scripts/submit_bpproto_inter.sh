#!/bin/bash
#SBATCH -J gra_282
#SBATCH -o gra_282.out
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1
#SBATCH -w node3
#SBATCH -c 2

echo "submitted from: "$SLURM_SUBMIT_DIR" on node:"$SLURM_SUBMIT_HOST
echo "Running on node:"$SLURM_JOB_NODELIST
echo "Allocate Gpu Units:"$CUDA_VISIBLE_DEVICES

nvidia-smi
CUDA_VISIBLE_DEVICES=1 python train_src.py --exp-suffix gra_282
