#!/bin/bash

for EPOCHS in {3..5}; do
  sbatch --export=EPOCHS=$EPOCHS train_CES.slurm
done
