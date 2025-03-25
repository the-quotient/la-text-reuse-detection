import json
import random
import os
from pathlib import Path

import torch
import torch.distributed as dist
from sentence_transformers import SentenceTransformer, util


class Sampler:
    def __init__(self, model_path, pairs_path, corpus_path):
        self.model = SentenceTransformer(
            os.path.expandvars(model_path)
        )
        self.pairs = self._load(os.path.expandvars(pairs_path))
        self.corpus = self._load(os.path.expandvars(corpus_path))
        self.rank, self.world_size = self.init_distributed()

    @staticmethod
    def init_distributed():
        if "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend="nccl")
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            return rank, world_size
        return 0, 1

    def sample_negatives_irrelevant(self, threshold=0.4, max_attempts=10):
        corpus_set = set(self.corpus)
        negatives = []

        # Partition the pairs among processes.
        if dist.is_initialized():
            pairs = self.pairs[self.rank::self.world_size]
        else:
            pairs = self.pairs

        for pair in pairs:
            s1, s2 = pair

            if s1 in corpus_set:
                original = s1
            elif s2 in corpus_set:
                original = s2
            else:
                continue

            for _ in range(max_attempts):
                candidate = random.choice(self.corpus)
                if candidate == original:
                    continue

                orig_emb = self.model.encode(
                    original, convert_to_tensor=True
                )
                cand_emb = self.model.encode(
                    candidate, convert_to_tensor=True
                )
                cosine_score = util.cos_sim(orig_emb, cand_emb).item()

                if cosine_score < threshold:
                    negatives.append((original, candidate))
                    break

        return negatives

    def save(self, pairs, label, output_file):
        output = []
        for sentence1, sentence2 in pairs:
            output.append({
                "sentence1": sentence1,
                "sentence2": sentence2,
                "label": label
            })

        # Append the rank to the output filename if in distributed mode.
        if dist.is_initialized():
            base, ext = os.path.splitext(output_file)
            output_file = f"{base}_rank{self.rank}{ext}"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=4)

    def _load(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        data = []

        if ext == '.jsonl':
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        data.append(record["sentence"])
        elif ext == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                for record in json_data:
                    sent1 = record.get("sentence1")
                    sent2 = record.get("sentence2")
                    if sent1 is not None and sent2 is not None:
                        data.append((sent1, sent2))
        else:
            raise ValueError(
                "Unsupported file extension. Only .json and .jsonl files are "
                "supported."
            )

        random.shuffle(data)
        return data
