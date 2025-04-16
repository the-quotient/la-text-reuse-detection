import argparse
import os
from benchmarking import BenchmarkRunner

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_folder")
    parser.add_argument("data_folder")
    parser.add_argument("result_folder")
    parser.add_argument("task_name")
    parser.add_argument("output_file")
    parser.add_argument("--label_map")
    args = parser.parse_args()

    runner = BenchmarkRunner(
        os.path.expandvars(args.model_folder),
        os.path.expandvars(args.data_folder),
        os.path.expandvars(args.result_folder)
    )

    for model_name in os.listdir(os.path.expandvars(args.model_folder)):
        model_path = os.path.join(os.path.expandvars(args.model_folder), model_name)
        if model_name.startswith("CES") and os.path.isdir(model_path):
            print(f"Benchmarking model: {model_name}")
            runner.benchmark_reranker(
                model_name, args.task_name, args.output_file, args.label_map
            )

if __name__ == "__main__":
    main()

