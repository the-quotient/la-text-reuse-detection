import json
import logging
import os
import time
from typing import List, Tuple

import openai
from openai import OpenAIError
from dotenv import load_dotenv

MAX_RETRIES = 10  
WAIT_TIME = 10 

class RequestRunner:

    def __init__(self, model: str, system_message: str, prompt: str,
                 temperature: float):
        self.model = model
        self.system_message = system_message
        self.prompt = prompt
        self.temperature = temperature

        load_dotenv() 
        openai.api_key = os.getenv("OPENAI_API_KEY")


    def process(self, batch: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
        retries = 0
        while retries < MAX_RETRIES:
            try:
                return self._process_batch(batch)
            except (json.JSONDecodeError, ValueError) as e:
                logging.warning(
                    f"Error processing batch: {e}. "
                    "Switching to single sentence processing."
                )
                return [self._process_batch([sentence_tuple])[0] 
                        for sentence_tuple in batch]
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
