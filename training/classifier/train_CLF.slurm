#!/bin/bash

#!/bin/bash
#SBATCH --job-name="CLF"
#SBATCH --output=logs/CLF_5.out
#SBATCH --error=logs/CLF_5.err
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

torchrun --nproc_per_node=6 train_classifier.py

