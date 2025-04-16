import argparse
import os
from benchmarking import BenchmarkRunner

def main():
    MODEL_FOLDER = "../../../scratch/tmp/p_krae02/models/v2/"
    DATA_FOLDER = "../../../scratch/tmp/p_krae02/data/eval-tasks-S/" 
    RESULT_FOLDER = "../../../scratch/tmp/p_krae02/eval/PL/" 

    TASK_NAME = "Ge"
    OUTPUT_FILE = "EVAL2-PL-S.json" 

    runner = BenchmarkRunner(MODEL_FOLDER, DATA_FOLDER, RESULT_FOLDER) 

    runner.benchmark_pipeline(
        "BEmargin_07_0",
        "CEPweight_09_0",
        "CESweight_05_0",
        TASK_NAME,
        OUTPUT_FILE
    )

if __name__ == "__main__":
    main()

