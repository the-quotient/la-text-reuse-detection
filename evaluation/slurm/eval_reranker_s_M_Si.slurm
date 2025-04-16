#!/bin/bash
#SBATCH --job-name=EVAL_CE_M_Si
#SBATCH --output=logs/EVAL_CE_M_Si.out
#SBATCH --error=logs/EVAL_CE_M_Si.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=p_krae02@uni-muenster.de
#SBATCH --partition=gpu4090,gpu3090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=0-10:00:00

module --force purge
module load palma/2020a

export LD_LIBRARY_PATH=/usr/lib:/usr/lib64:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_FOLDER=$WORK/models/v2
DATA_FOLDER=$WORK/data/eval-tasks-M/
RESULT_FOLDER=$WORK/eval/
TASK_NAME=Si
OUTPUT_FILE=eval_CE_M_Si.json

srun python eval_reranker_s.py $MODEL_FOLDER $DATA_FOLDER $RESULT_FOLDER \
     $TASK_NAME $OUTPUT_FILE
