import argparse
import json
import random
from typing import List

from api_requests import RequestRunner


MODEL = "gpt-4o"


modifications = {
    1: "add a few introductory words to the sentence. ",
    2: "add a few concluding words to the sentence. ",
    3: "enclose the sentence with a few words. "
}

def get_prompt(style: int):
    return (
        "Change the sentence in the following way: "
        f"Considering and adapting to the context, {modifications[style]}"
        "If the sentence already has an introduction or conclusion, "
        "change it to something else. Return the whole sentence."
    )


def create_fuzzy_quotes(
    input_file: str,
    output_file: str,
    sample_size: int,
    batch_size: int
):

    fuzzy_quoter = RequestRunner(MODEL, "Latin")
    batches = fuzzy_quoter.load(input_file, sample_size, batch_size)

    result = []
    for batch in batches:
        fuzzy_quotes = fuzzy_quoter.process(
            batch,
            random.uniform(0.8, 1.2),
            get_prompt(random.randint(1, 3))
        )

        for (i, sent1), (j, sent2) in zip(batch, fuzzy_quotes):
            if i == j:
                result.append((sent1, sent2))

    fuzzy_quoter.save(result, "fuzzy_quote", output_file)




def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("output_file")
    parser.add_argument("sample_size", type=int)
    parser.add_argument("batch_size", type=int)
    args = parser.parse_args()

    create_fuzzy_quotes(
        args.input_file, 
        args.output_file, 
        args.sample_size, 
        args.batch_size
    )

if __name__ == "__main__":
    main()

