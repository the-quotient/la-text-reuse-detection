#!/bin/bash

#SBATCH --job-name=CE-MLM
#SBATCH --output=CE-MLM.out
#SBATCH --error=CE-MLM.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=p_krae02@uni-muenster.de

#SBATCH --partition=gpu4090,gpu3090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:6
#SBATCH --time=7-00:00:00

module --force purge
module load palma/2020a

export LD_LIBRARY_PATH=/usr/lib:/usr/lib64:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=6

torchrun --nproc_per_node=6 mlm.py \
  CE-MLM \
  $WORK/models/LaBERTa/ \
  $WORK/models/ \
  $WORK/data/mlm/mlm_train.jsonl \
  $WORK/data/mlm/mlm_dev.jsonl
