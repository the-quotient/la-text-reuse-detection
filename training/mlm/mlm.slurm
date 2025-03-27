#!/bin/bash

#SBATCH --job-name=CE-MLM
#SBATCH --output=logs/CE-MLM.out
#SBATCH --error=logs/CE-MLM.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=p_krae02@uni-muenster.de

#SBATCH --partition=gpu4090,gpu3090,gpu2080,gputitanrtx,gpua100,gpuhgx
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --time=1-00:00:00

module --force purge
module load palma/2020a

export LD_LIBRARY_PATH=/usr/lib:/usr/lib64:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=4

torchrun --nproc_per_node=4 mlm.py \
  CE-MLM \
  $WORK/models/LaBERTa/ \
  $WORK/models/ \
  $WORK/data/mlm/corpus.jsonl \
