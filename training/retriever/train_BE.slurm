#!/bin/bash

MARGINS=(0.2 0.5 1.0 1.5 2.0)

MODEL_PATH="$WORK/models/SPhilBERTa/"
DATA_FILE="$WORK/data/retriever/training_retriever.json"
OUTPUT_DIR="$WORK/models/"

SEQ_LENGTH=128
BATCH_SIZE=32
EPOCHS=5

for MARGIN in "${MARGINS[@]}"; do
    VERSION="margin_${MARGIN//./_}"

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name="BE_${VERSION}"
#SBATCH --output=logs/BE_${VERSION}.out
#SBATCH --error=logs/BE_${VERSION}.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=p_krae02@uni-muenster.de
#SBATCH --partition=gpu4090,gpu3090,gpu2080
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:6
#SBATCH --time=1-00:00:00

module --force purge
module load palma/2020a

export LD_LIBRARY_PATH=/usr/lib:/usr/lib64:\$LD_LIBRARY_PATH
export OMP_NUM_THREADS=6

torchrun --nproc_per_node=6 train_retriever.py \\
    --pretrained_model_path "$MODEL_PATH" \\
    --data_file "$DATA_FILE" \\
    --output_base_path "$OUTPUT_DIR" \\
    --version "$VERSION" \\
    --train_batch_size $BATCH_SIZE \\
    --num_epochs $EPOCHS \\
    --max_seq_length $SEQ_LENGTH \\
    --triplet_margin $MARGIN
EOF

done

