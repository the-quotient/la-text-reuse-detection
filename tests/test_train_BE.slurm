#!/bin/bash

#SBATCH --job-name="TEST-BE"
#SBATCH --output=logs/TEST-BE_%j.out
#SBATCH --error=logs/TEST-BE_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=p_krae02@uni-muenster.de

#SBATCH --partition=gpuexpress
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --time=0-00:10:00

MODEL_PATH="$WORK/models/SPhilBERTa/"
DATA_FILE="$WORK/data/test/test_training_retriever.json"
OUTPUT_DIR="$WORK/models/tests"

SEQ_LENGTH=128
BATCH_SIZE=32
EPOCHS=5

TRIPLET_MARGIN=1.0
VERSION="v1"

module --force purge
module load palma/2020a

export LD_LIBRARY_PATH=/usr/lib:/usr/lib64:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=2

torchrun --nproc_per_node=2 train_retriever.py \
    --pretrained_model_path "$MODEL_PATH" \
    --data_file "$DATA_FILE" \
    --output_base_path "$OUTPUT_DIR" \
    --version "$VERSION" \
    --train_batch_size $BATCH_SIZE \
    --num_epochs $EPOCHS \
    --max_seq_length $SEQ_LENGTH \
    --triplet_margin $TRIPLET_MARGIN
