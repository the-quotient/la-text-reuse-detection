import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import torch
import torch.distributed as dist

from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments
)


# Hyperparameters
per_device_train_batch_size = 16
num_train_epochs = 15
use_fp16 = True
max_length = 256
mlm_prob = 0.15
weight_decay = 0.01


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(log_file, rank):
    if rank == 0:
        logging.basicConfig(
            filename=log_file,
            format="%(asctime)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.INFO,
        )
    else:
        logging.basicConfig(level=logging.ERROR)
    return logging.getLogger(__name__)


def load(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        sentences = [
            json.loads(line)["sentence"] for line in f if line.strip()
        ]
    random.shuffle(sentences)
    split_index = int(len(sentences) * 0.9)
    train_sentences = sentences[:split_index]
    dev_sentences = sentences[split_index:]
    return train_sentences, dev_sentences


class TokenizedSentencesDataset:
    def __init__(
            self, sentences, tokenizer, max_length, 
            cache_tokenization=False
        ):
        self.tokenizer = tokenizer
        self.sentences = sentences
        self.max_length = max_length
        self.cache_tokenization = cache_tokenization

    def __getitem__(self, index):
        if self.cache_tokenization:
            if isinstance(self.sentences[index], str):
                self.sentences[index] = self.tokenizer(
                    self.sentences[index],
                    add_special_tokens=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_special_tokens_mask=True,
                    return_attention_mask=True
                )
            return self.sentences[index]
        return self.tokenizer(
            self.sentences[index],
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            return_special_tokens_mask=True,
            return_attention_mask=True
        )

    def __len__(self):
        return len(self.sentences)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str)
    parser.add_argument("pretrained_model_name", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("data_file", type=str)
    args = parser.parse_args()

    set_seed(42)

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    device = torch.device(
        f"cuda:{local_rank}" 
        if (local_rank >= 0 and torch.cuda.is_available())
        else "cpu"
    )

    if local_rank >= 0:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank()
    else:
        rank = 0

    output_dir = Path(args.output_dir) / f"{args.model_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    os.makedirs("logs", exist_ok=True)
    logger = setup_logging(f"logs/{args.model_name}.log", local_rank)
    logger.info("Starting training...")

    logger.info("Loading model and tokenizer...")
    model = AutoModelForMaskedLM.from_pretrained(
        args.pretrained_model_name
    )
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name)

    logger.info("Loading training and dev data...")
    train_sentences, dev_sentences = load(args.data_file)
    logger.info("Loaded %s training sentences.", len(train_sentences))
    logger.info("Loaded %s dev sentences.", len(dev_sentences))

    train_dataset = TokenizedSentencesDataset(train_sentences, tokenizer, max_length)
    dev_dataset = TokenizedSentencesDataset(
        dev_sentences, tokenizer, max_length, cache_tokenization=True
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=mlm_prob
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=per_device_train_batch_size,
        fp16=True,
        local_rank=local_rank,
        ddp_find_unused_parameters=False,
        weight_decay=weight_decay,
        load_best_model_at_end = True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    if local_rank == 0:
        logger.info("Saving tokenizer to: %s", output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info("Saving model to: %s", output_dir)
        model.save_pretrained(output_dir)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
