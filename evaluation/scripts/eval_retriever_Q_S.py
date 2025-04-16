import argparse
import os
from benchmarking import BenchmarkRunner

def main():
    MODEL_FOLDER = "../../../scratch/tmp/p_krae02/models/v2/"

    DATA_FOLDER = "../../../scratch/tmp/p_krae02/data/eval-tasks-S/" 
    RESULT_FOLDER = "../../../scratch/tmp/p_krae02/eval/BE/" 


    TASK_NAME = "Ge"
    LABEL_MAP = "Qu"
    OUTPUT_FILE = "EVAL-BE-QS.json" 

    runner = BenchmarkRunner(MODEL_FOLDER, DATA_FOLDER, RESULT_FOLDER) 

    for model_name in os.listdir(MODEL_FOLDER):
        model_path = os.path.join(MODEL_FOLDER, model_name)
        if (model_name.startswith("BE") or model_name.startswith("SPhilBERTa")) and os.path.isdir(model_path):
            print(f"Benchmarking model: {model_name}")
            runner.benchmark_retriever(
                model_name, TASK_NAME, OUTPUT_FILE, LABEL_MAP
            )

if __name__ == "__main__":
    main()
