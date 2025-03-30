#!/bin/bash

POS_WEIGHTS=(0.5 1.0 2.0 5.0 10.0)

MODEL_NAME="$WORK/models/CE-MLM/"
DATASET_NAME="$WORK/data/training_reranker_s.json"
OUTPUT_BASE_DIR="$WORK/models/v2/"

SEQ_LENGTH=256
BATCH_SIZE=32
EPOCHS=5

LABEL_MAPPING="s"

for POS_WEIGHT in "${POS_WEIGHTS[@]}"; do
    VERSION="posweight_${POS_WEIGHT//./_}"

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name="CE_${VERSION}"
#SBATCH --output=logs/CE_${VERSION}.out
#SBATCH --error=logs/CE_${VERSION}.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=p_krae02@uni-muenster.de
#SBATCH --partition=gpu4090,gpu3090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00

module --force purge
module load palma/2020a

export LD_LIBRARY_PATH=/usr/lib:/usr/lib64:\$LD_LIBRARY_PATH
export OMP_NUM_THREADS=4

torchrun --nproc_per_node=6 train_reranker.py \\
    --model_name "\$MODEL_NAME" \\
    --dataset_name "\$DATASET_NAME" \\
    --output_dir "\${OUTPUT_BASE_DIR}/\${VERSION}" \\
    --epochs \$EPOCHS \\
    --batch_size \$BATCH_SIZE \\
    --learning_rate 5e-5 \\
    --max_seq_length \$SEQ_LENGTH \\
    --pos_weight \$POS_WEIGHT \\
    --label_mapping "\$LABEL_MAPPING" \\
    --fp16
EOF

done

