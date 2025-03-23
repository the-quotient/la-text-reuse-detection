import argparse
import json
import random
from typing import List

from api_requests import RequestRunner


MODEL = "gpt-4o"


def get_prompt_paraphrase():
    return (
        "Translate into English and write a paraphrase that preserves the original meaning."
        "Use different grammatical constructions and different words as much as possible."
    ) 


def create_paraphrases(
    input_file: str, 
    output_file: str, 
    sample_size: int, 
    batch_size: int
):

    paraphraser = RequestRunner(MODEL, "English")
    batches = paraphraser.load(input_file, sample_size, batch_size)
    translator = RequestRunner(MODEL, "Latin", "English -> Latin") 

    result = []
    for batch in batches:
        english_sentences = paraphraser.process(
            batch, 
            random.uniform(0.8, 1.2),
            get_prompt_paraphrase()
        )
        latin_sentences = translator.process(english_sentences, random.uniform(0.5, 1.0))

        for (i, sent1), (j, sent2) in zip(batch, latin_sentences):
            if i == j:
                result.append((sent1, sent2))

    translator.save(result, "paraphrase", output_file)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("output_file")
    parser.add_argument("sample_size", type=int)
    parser.add_argument("batch_size", type=int)
    args = parser.parse_args()

    create_paraphrases(
        args.input_file, 
        args.output_file, 
        args.sample_size, 
        args.batch_size
    )

if __name__ == "__main__":
    main()

