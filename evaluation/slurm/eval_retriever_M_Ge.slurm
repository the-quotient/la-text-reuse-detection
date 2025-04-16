#!/bin/bash
#SBATCH --job-name=EVAL_BE_M_Ge
#SBATCH --output=logs/EVAL_BE_M_Ge.out
#SBATCH --error=logs/EVAL_BE_M_Ge.err
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

MODEL_FOLDER=$WORK/models/v2
DATA_FOLDER=$WORK/data/eval-tasks-M/
RESULT_FOLDER=$WORK/eval/
TASK_NAME=Ge
OUTPUT_FILE=eval_BE_M_Ge.json

srun python eval_retriever.py $MODEL_FOLDER $DATA_FOLDER $RESULT_FOLDER \
     $TASK_NAME $OUTPUT_FILE
