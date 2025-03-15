#!/bin/bash

#SBATCH --job-name=TEST-MLM
#SBATCH --output=TEST-MLM.out
#SBATCH --error=TEST-MLM.err
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

torchrun --nproc_per_node=2 mlm.py \
  TEST-MLM \
  $WORK/models/LaBERTa-base/ \
  $WORK/models/tests/ \
  $WORK/data/test/mlm_test_train.jsonl \
  $WORK/data/test/mlm_test_dev.jsonl
