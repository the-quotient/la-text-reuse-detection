#!/bin/bash

POS_WEIGHTS=(0.5 1.0 2.0 5.0 10.0)

MODEL_NAME="$WORK/models/CE-MLM/"
DATASET_NAME="$WORK/data/training_reranker_p_v2.json"
OUTPUT_BASE_DIR="$WORK/models/v2/"

SEQ_LENGTH=256
BATCH_SIZE=32
EPOCHS=5

LABEL_MAPPING="p"

for POS_WEIGHT in "${POS_WEIGHTS[@]}"; do
    VERSION="weight_${POS_WEIGHT//./_}"

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name="CEP_${VERSION}"
#SBATCH --output=logs/CEP_${VERSION}.out
#SBATCH --error=logs/CEP_${VERSION}.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=p_krae02@uni-muenster.de
#SBATCH --partition=gpu4090,gpu3090
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

torchrun --nproc_per_node=6 train_reranker.py \\
    --model_name "$MODEL_NAME" \\
    --dataset_name "$DATASET_NAME" \\
    --output_dir "${OUTPUT_BASE_DIR}/CEP${VERSION}" \\
    --epochs $EPOCHS \\
    --batch_size $BATCH_SIZE \\
    --learning_rate 5e-5 \\
    --max_seq_length $SEQ_LENGTH \\
    --pos_weight $POS_WEIGHT \\
    --label_mapping "$LABEL_MAPPING" \\
    --fp16
EOF

done

