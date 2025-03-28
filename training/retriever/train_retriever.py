import os
import random
import logging
import argparse
import json

import numpy as np
import torch
import torch.distributed as dist
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    losses,
    InputExample,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load(data_file):
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    all_samples = [
        {"anchor": entry["anchor"], "positive": entry["positive"], "negative": entry["negative"]}
        for entry in data
    ]
    random.shuffle(all_samples)
    train_data = Dataset.from_list(all_samples)
    return train_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", required=True)
    parser.add_argument("--data_file", required=True)
    parser.add_argument("--output_base_path", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--train_batch_size", type=int, required=True)
    parser.add_argument("--num_epochs", type=int, required=True)
    parser.add_argument("--max_seq_length", type=int, required=True)
    parser.add_argument("--triplet_margin", type=float, default=1.0)
    args = parser.parse_args()

    set_seed(42)

    train_dataset = load(args.data_file)

    model = SentenceTransformer(args.pretrained_model_path)
    model.max_seq_length = args.max_seq_length

    loss_function = losses.TripletLoss(model=model, triplet_margin=args.triplet_margin)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=os.path.join(args.output_base_path, f"BiEncoder{args.version}"),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.train_batch_size,
        warmup_ratio=0.1,
        fp16=True,
        dataloader_drop_last=True
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=loss_function,
    )

    trainer.train()


if __name__ == "__main__":
    main()

