#!/bin/bash

#SBATCH --job-name=SAMPLE-NEGATIVES
#SBATCH --output=SAMPLE-NEGATIVES.out
#SBATCH --error=SAMPLE-NEGATIVES.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=p_krae02@uni-muenster.de

#SBATCH --partition=gpuexpress
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --time=0-00:10:00

module --force purge
module load palma/2020a

export LD_LIBRARY_PATH=/usr/lib:/usr/lib64:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=4

torchrun --nproc_per_node=2 server_sampling.py \
  $WORK/models/SPhilBERTa \
  $WORK/data/test_paraphrases.json \
  $WORK/data/corpus.jsonl \
  $WORK/data/irrelevant_for_paraphrases.json \
  irrelevant
