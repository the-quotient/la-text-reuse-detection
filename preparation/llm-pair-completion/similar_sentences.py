import argparse
import json
import random
from typing import List

from api_requests import RequestRunner

MODEL = "gpt-4o"


def get_prompt_similar_sentence(
    token_count: int, 
    if_introduction: bool, 
    transformation_type: int
):

    prompt = (
        "Identify the central thought or message of the sentence and "
        "transform this thought according to the following task: \n"
    )

    transformation_options = {
        1: "Change the perspective.",
        2: "Spin the thought further.",
        3: "Apply the thought to a different situation.",
        4: "Exaggerate the thought for dramatic or humorous effect.",
        5: "Combine the thought with a seemingly unrelated idea.",
        6: "Introduce the thought and reject it."
    }

    if transformation_type in transformation_options:
        prompt += transformation_options[transformation_type]

    prompt += ("\nThink of three to five words as introduction into the thought."
               if if_introduction else "")

    prompt += (
        f"\nThen, integrate the transformed thought into a single new English sentence "
        f"of about {token_count} tokens. Use 16th century thinking and language "
        "in a theological context.\n"
    )

    return prompt


def create_similar_sentences(
    input_file: str, 
    output_file: str, 
    sample_size: int, 
    batch_size: int
): 

    creator = RequestRunner(MODEL, "English")
    batches = creator.load(input_file, sample_size, batch_size) 
    translator = RequestRunner(MODEL, "Latin", "English -> Latin")

    numbers = [1,2,3,4,5,6]
    weights = [3,3,3,2,2,1]

    result = []
    for batch in batches:
        english_sentences = creator.process(
            batch, 
            random.uniform(0.8, 1.2),
            get_prompt_similar_sentence(
                random.randint(10, 40),
                random.random() > 0.75,
                random.choices(numbers, weights=weights, k=1)[0]
            )
        )
        latin_sentences = translator.process(english_sentences, random.uniform(0.5,1.0))

        for (i, sent1), (j, sent2) in zip(batch, latin_sentences):
            if i == j:
                result.append((sent1, sent2))

    translator.save(result, "similar_sentence", output_file) 


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("output_file")
    parser.add_argument("sample_size", type=int)
    parser.add_argument("batch_size", type=int)
    args = parser.parse_args()

    create_similar_sentences(
        args.input_file, 
        args.output_file, 
        args.sample_size, 
        args.batch_size
    )

if __name__ == "__main__":
    main()

