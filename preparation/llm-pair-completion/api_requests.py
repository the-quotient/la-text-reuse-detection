import json
import logging
import os
import time
import random
from typing import List, Tuple

import openai
from openai import OpenAIError
from dotenv import load_dotenv

MAX_RETRIES = 10
WAIT_TIME = 10


examples_la = (
    "Example response:\n"
    "[\n"
    "  {\"index\": 0, \"processed_sentence\": \"Nihil enim nobis dulcius, quam de "
    "aliorum erratis loqui, nec quicquam hic periculi suspicamur.\"},\n"
    "  {\"index\": 1, \"processed_sentence\": \"Praetendit enim plerumque hypocrisis "
    "nostra, quasi ex compassione peccatum fratris referat, cum uere id non nisi aut "
    "ex odio, aut ad acquirendam sibi laudem proficiscatur.\"},\n"
    "  {\"index\": 2, \"processed_sentence\": \"Hypocrita, si fratri compateris, quid "
    "opus est, ut peccatum eius aliis pandas, cur non famae fratris consulis?\"},\n"
    "  {\"index\": 3, \"processed_sentence\": \"Docet diuus Iacobus, qui in uerbo non "
    "offendit, hic perfectus est, id quod potissimum uerificatur in hoc casu.\"}\n"
    "]"
)

examples_en = (
    "Example response:\n"
    "[\n"
    "  {\"index\": 0, \"processed_sentence\": \"Nothing is sweeter to us than to "
    "speak of the errors of others, and we suspect no danger here.\"},\n"
    "  {\"index\": 1, \"processed_sentence\": \"Our hypocrisy often pretends to report "
    "a brother’s sin out of compassion, when in truth it stems either from hatred or "
    "from a desire to gain praise for oneself.\"},\n"
    "  {\"index\": 2, \"processed_sentence\": \"Hypocrite, if you truly pity your "
    "brother, why must you spread his sin to others? Why not care for his reputation?\"},\n"
    "  {\"index\": 3, \"processed_sentence\": \"Saint James teaches that whoever does "
    "not offend in word is perfect—and this is especially true in this case.\"}\n"
    "]"
)


class RequestRunner:

    def __init__(self, model: str, language: str, translation=None):
        self.model = model
        self.system_message = self._get_system_message(language)
        self.prompt = translation and self._get_prompt_translation(translation)
        self.temperature = 0.5

        load_dotenv() 
        openai.api_key = os.getenv("OPENAI_API_KEY")


    def process(
        self, 
        batch: List[List[Tuple[int, str]]], 
        temperature: float,
        prompt=None
    ) -> List[Tuple[int, str]]:

        self.temperature = temperature
        if prompt is not None:
            self.prompt = prompt 

        retries = 0
        while retries < MAX_RETRIES:
            try:
                return self._process_batch(batch)
            except (json.JSONDecodeError, ValueError) as e:
                logging.warning(
                    f"Error processing batch: {e}. "
                    "Switching to single sentence processing."
                )
                result = []
                for sentence in batch:
                    try:
                        result.append(self._process_batch([sentence])[0])
                    except Exception as e:
                        logging.warning(f"Error processing sentence: {e}")
                return result
            except OpenAIError as e:
                logging.warning(f"OpenAI API error: {e}")
                retries += 1
                time.sleep(WAIT_TIME)
            except Exception as e:
                logging.warning(f"Unexpected error: {e}")
                retries += 1
                time.sleep(WAIT_TIME)
        raise RuntimeError("Maximum retries reached. API is not responding.")


    def _process_batch(self, batch: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
        response_str = self._request(self._build_json_input(batch))
        response = json.loads(response_str)
        if len(response) != len(batch):
            raise ValueError("Mismatch between input and output sentence counts.")
        results = []
        for (input_index, _), item in zip(batch, response):
            if item.get('index') != input_index:
                raise ValueError(f"Index mismatch for index {input_index}")
            processed = item.get('processed_sentence', "").strip()
            results.append((input_index, processed))
        return results


    def _build_json_input(self, batch: List[Tuple[int, str]]) -> str:
        return json.dumps({
            "instructions": self.prompt,
            "sentences": [
                {"index": index, "sentence": sentence}
                for index, sentence in batch
            ]
        }, ensure_ascii=False, indent=2)


    def _request(self, content: str) -> str:
        return openai.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": content}
            ]
        ).choices[0].message.content


    def _get_system_message(self, language):
        return (
            f"You are an Early Modern scholar from the 16th century specialized in "
            f"reading and writing {language} theological texts.\n"
            "You will process a JSON array of indexed sentences, executing the given "
            "instructions for each sentence. For each sentence you will produce exactly "
            "one sentence. You will return a JSON array of processed sentences, "
            "maintaining the index. Do not enclose it in Markdown!\n"
        ) + (examples_la if language == "Latin" else examples_en)


    def _get_prompt_translation(self, translation_direction: str):

        if translation_direction == "English -> Latin":
            return (
                "Translate into a single sentence in 16th century theological Latin." 
                "Check for grammatical correctness and refine if necessary." 
            )
        else: 
            raise ValueError("Direction not yet supported!") 


    # Loads and batches the data 
    def load(
        self, 
        file_path: str, 
        sample_size: int, 
        batch_size: int
    ) -> List[List[Tuple[int, str]]]:

        with open(file_path, 'r', encoding='utf-8') as f:
            sentences = [json.loads(line)["sentence"] for line in f if line.strip()]

        if sample_size > len(sentences):
            raise ValueError("Sample size exceeds the total number of sentences in the file.")

        sampled_sentences = random.sample(sentences, sample_size)
        random.shuffle(sampled_sentences)

        # Assign unique global index to each sampled sentence
        indexed_sentences = list(enumerate(sampled_sentences))

        # Split into batches
        batches = [
            indexed_sentences[i:i + batch_size]
            for i in range(0, len(indexed_sentences), batch_size)
        ]

        return batches


    # Saves (original, generated, label) pairs 
    def save(self, tuples_list: List[Tuple[str]], label: str, output_file: str):

        processed = [
            tup if random.random() < 0.5 else tup[::-1]
            for tup in tuples_list
        ]

        output_data = []
        for tup in processed:
            sentence1, sentence2 = tup
            output_data.append({
                "sentence1": sentence1,
                "sentence2": sentence2,
                "label": label
            })

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4)

        print(output_file)

