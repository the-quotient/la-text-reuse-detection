#!/bin/bash

#SBATCH --job-name=TEST-CE-TRAINING
#SBATCH --output=TEST-CE-TRAINING.out
#SBATCH --error=TEST-CE-TRAINING.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=p_krae02@uni-muenster.de

#SBATCH --partition=gpuexpress
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --time=0-00:10:00

module --force purge
module load palma/2020a

export LD_LIBRARY_PATH=/usr/lib:/usr/lib64:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=4

torchrun --nproc_per_node=2 train_reranker.py \
  "$WORK/models/CE-MLM/" \
  "$WORK/data/test/test_training.json" \
  "$WORK/models/tests/" \
  "p" \
  "test" \
  0.9 \
  16 \
  3

