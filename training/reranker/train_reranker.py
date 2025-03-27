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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets import Dataset

from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder,
    losses,
    InputExample,
)
from sentence_transformers.models import Transformer, Pooling
from sentence_transformers.cross_encoder.evaluation import (
    CEF1Evaluator
)
from sentence_transformers.evaluation import (
    EmbeddingSimilarityEvaluator,
    SequentialEvaluator
)
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

from transformers import AutoTokenizer


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
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


def prepare_data(data_file, ce_label, logger):
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
    split_index = int(0.9 * len(all_samples))
    return all_samples[:split_index], all_samples[split_index:]


def train_cross_encoder(args, ce_label, rank, logger):

    num_labels = 2

    train_samples, dev_samples = prepare_data(
        args.data_file, ce_label, logger
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
        max_length=args.max_seq_length
    )

    if dist.is_initialized():
        model.model = DDP(model.model, device_ids=[rank])

    if dist.is_initialized():
        train_sampler = DistributedSampler(
            train_samples,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True
        )
        train_dataloader = DataLoader(
            train_samples,
            batch_size=args.train_batch_size,
            sampler=train_sampler
        )
    else:
        train_dataloader = DataLoader(
            train_samples,
            batch_size=args.train_batch_size,
            shuffle=True
        )

    dev_f1_evaluator = CEF1Evaluator.from_input_examples(
        dev_samples, name="dev-f1"
    )

    train_steps_per_epoch = len(train_dataloader)
    total_steps = train_steps_per_epoch * args.num_epochs
    total_warmup_steps = int(0.1 * total_steps)
    warmup_per_epoch = total_warmup_steps // args.num_epochs

    output_path = os.path.join(
        args.output_base_path,
        f"CE{args.label.upper()}-{args.version}"
    )

    for epoch in range(args.num_epochs):
        if dist.is_initialized():
            train_dataloader.sampler.set_epoch(epoch)

        model.fit(
            train_dataloader=train_dataloader,
            evaluator=dev_f1_evaluator,
            epochs=1,
            warmup_steps=warmup_per_epoch,
            output_path=None,
            loss_fct=loss_function,
            use_amp=True
        )
        if rank == 0:
            logger.info(f"Finished epoch {epoch+1}/{args.num_epochs}")

    if rank == 0:
        if isinstance(model.model, DDP):
            model.model.module.save_pretrained(output_path)
        else:
            model.model.save_pretrained(output_path)

        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)
        tokenizer.save_pretrained(output_path)

        logger.info(f"Final model and tokenizer saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", required=True)
    parser.add_argument("--data_file", required=True)
    parser.add_argument("--output_base_path", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--train_batch_size", type=int, required=True)
    parser.add_argument("--num_epochs", type=int, required=True)
    parser.add_argument("--max_seq_length", type=int, required=True)
    args = parser.parse_args()

    set_seed(42)

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
        raise ValueError("Only p for paraphrase and s for similar_sentences supported.")

    logger = setup_logging(f"logs/CE{args.label.upper()}-{args.version}.log", rank)
    logger.info("Starting training...")

    train_cross_encoder(args, ce_label, rank, logger)

    if local_rank >= 0:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

