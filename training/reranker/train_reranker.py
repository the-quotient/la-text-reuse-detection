import os
import random
import logging
import argparse
import json
from datetime import datetime
from collections import Counter

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from datasets import Dataset

from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder,
    losses,
    InputExample,
)
from sentence_transformers.models import Transformer, Pooling
from sentence_transformers.cross_encoder.evaluation import (
    CEF1Evaluator, 
    CESoftmaxAccuracyEvaluator
)
from sentence_transformers.evaluation import (
    EmbeddingSimilarityEvaluator,
    SequentialEvaluator
)
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments


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


def prepare_data(data_file, ce_label, train_ratio, logger):
    """
    Loads data from a JSON file with structure sentence1, sentence2, label.
    Maps labels: "irrelevant" -> 0, others -> 1.
    Shuffles the data and splits it into training and validation sets.
    """
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    all_samples = []
    for i, entry in enumerate(data):
        sentence1 = entry["sentence1"]
        sentence2 = entry["sentence2"]
        raw_label = entry["label"]
        mapped_label = 1 if raw_label.lower() == ce_label else 0
        all_samples.append(
            InputExample(guid=str(i), texts=[sentence1, sentence2], 
                         label=mapped_label)
        )
    random.shuffle(all_samples)
    split_index = int(train_ratio * len(all_samples))
    return all_samples[:split_index], all_samples[split_index:]


def train_cross_encoder(args, ce_label, rank, logger):

    num_labels = 2

    train_samples, dev_samples = prepare_data(
        args.data_file, ce_label, args.train_ratio, logger
    )

    device = torch.device(
        f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"[Rank {rank}] Using device {device}")

    loss_function = torch.nn.CrossEntropyLoss()

    model = CrossEncoder(
        args.pretrained_model_path,
        num_labels=num_labels,
        device=device,
        max_length=512
    )

    train_dataloader = DataLoader(
        train_samples, batch_size=args.train_batch_size, shuffle=True
    )
    dev_f1_evaluator = CEF1Evaluator.from_input_examples(
        dev_samples, name="dev-f1"
    )

    output_path = os.path.join(
        args.output_base_path,
        f"SF-{args.label.upper()}-{args.version}-"
        f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    )

    warmup_steps = int(0.1 * len(train_dataloader) * args.num_epochs)

    model.fit(
        train_dataloader=train_dataloader,
        evaluator=dev_f1_evaluator,
        epochs=args.num_epochs,
        warmup_steps=warmup_steps,
        output_path=output_path,
        loss_fct=loss_function
    )
    logger.info(f"Model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pretrained_model_path")
    parser.add_argument("data_file")
    parser.add_argument("output_base_path")
    parser.add_argument("label")
    parser.add_argument("version", default="")
    parser.add_argument("train_ratio", type=float, default=0.9)
    parser.add_argument("train_batch_size", type=int)
    parser.add_argument("num_epochs", type=int)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    device = torch.device(
        f"cuda:{local_rank}" if (local_rank >= 0 and torch.cuda.is_available())
        else "cpu"
    )

    if local_rank >= 0:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank()
    else:
        rank = 0

    if args.label == "p":
        ce_label = "paraphrase"
    elif args.label == "s":
        ce_label = "similar_sentence"
    else:
        raise Error()

    logger = setup_logging(f"CE{ce_label}{args.version}", rank)
    logger.info("Starting training...")

    train_cross_encoder(args, ce_label, rank, logger)

    if local_rank >= 0:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

