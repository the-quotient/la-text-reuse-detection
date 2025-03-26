import argparse
from sampling import Sampler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("pairs_path")
    parser.add_argument("corpus_path")
    parser.add_argument("output_file")
    parser.add_argument("label")

    args = parser.parse_args()

    sampler = Sampler(
        model_path=args.model_path,
        pairs_path=args.pairs_path,
        corpus_path=args.corpus_path
    )

    negatives = sampler.sample_negatives_irrelevant()
    sampler.save(negatives, label=args.label, output_file=args.output_file)

if __name__ == "__main__":
    main()
