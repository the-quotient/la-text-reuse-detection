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
    parser.add_argument("--config")
    args = parser.parse_args()

    runner = BenchmarkRunner(
        os.path.expandvars(args.model_folder),
        os.path.expandvars(args.data_folder),
        os.path.expandvars(args.result_folder)
    )

    for model_name in os.listdir(os.path.expandvars(args.model_folder)):
        model_path = os.path.join(os.path.expandvars(args.model_folder), model_name)
        if (model_name.startswith("BE") or model_name.startswith("SPhilBERTa")) and os.path.isdir(model_path):
            print(f"Benchmarking model: {model_name}")
            if args.config:
                runner.benchmark_retriever(
                    model_name, args.task_name, args.output_file, config_label=args.config
                )
            else:
                runner.benchmark_retriever(
                    model_name, args.task_name, args.output_file
                )

if __name__ == "__main__":
    main()

