import json
import os
import re
import sys
import warnings
from collections import defaultdict
from itertools import product

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from sentence_transformers import SentenceTransformer, CrossEncoder

sys.path.append(os.path.abspath('..'))
from app import Retriever, Reranker, run_pipeline


LABELS = ["irrelevant", "similar_sentence", "paraphrase", "fuzzy_quote", "quote"]

LABEL_MAP = {label: idx for idx, label in enumerate(LABELS)}

def make_binary_map(target_labels):
    return {label: int(label in target_labels) for label in LABELS}

TASK_LABEL_MAP = {
    "Ge": LABEL_MAP,
    "Qu": make_binary_map({"quote"}),
    "Fu": make_binary_map({"fuzzy_quote"}),
    "Pa": make_binary_map({"paraphrase"}),
    "Si": make_binary_map({"similar_sentence"}),
}

RETRIEVAL_CONFIGS_TEST = [
    {"k": 3, "threshold": 0.7}
]

RETRIEVAL_CONFIGS_GE = [
    {"k": k, "threshold": threshold}
    for k in [3, 5, 7]
    for threshold in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
]

RETRIEVAL_CONFIGS_QU = [
    {"k": 1, "threshold": threshold}
    for threshold in [0.95, 0.96, 0.97, 0.98]
]

RERANKING_CONFIGS = [
        threshold for threshold in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
]

PIPELINE_CONFIGS = [
    {
        "k": k,
        "quote_threshold": quote_threshold,
        "retrieval_threshold": retrieval_threshold,
        "reranking_p_threshold": reranking_p_threshold,
        "reranking_s_threshold": reranking_s_threshold
    }
    for k in [3]
    for quote_threshold in [0.99]
    for retrieval_threshold in [0.65]
    for reranking_p_threshold in [0.1, 0.15, 0.2]
    for reranking_s_threshold in [0.4]
]

class BenchmarkRunner:

    def __init__(self, model_folder, task_folder, result_folder):
        self.model_folder = model_folder
        self.task_folder = task_folder
        self.result_folder = result_folder


    def benchmark_retriever(self, model_name, task_name, output_file, label_map=None):

        tasks = self._load_tasks(task_name)
        dataframes = self._load_data(tasks, label_map)

        model_path = os.path.join(self.model_folder, model_name)

        tagged_results = []
        for df, sample_name in zip(dataframes, tasks):
            positive_map = defaultdict(set)
            for _, row in df.iterrows():
                if row['label'] == 1:
                    positive_map[row['sentence1']].add(row['sentence2'])

            queries = list(df['sentence1'].unique())
            candidates = list(df['sentence2'].unique())

            retriever = Retriever(model_path, queries, candidates=candidates)

            config_results = []

            if label_map == "Qu":
                configs = RETRIEVAL_CONFIGS_QU
            else:
                configs = RETRIEVAL_CONFIGS_GE

            for config in configs:
                k = config.get("k")
                threshold = config.get("threshold")
                results = retriever.retrieve(k, threshold)

                positive_hits = 0
                total_positive = len([q for q in queries if q in positive_map])
                precision_list = []
                f1_list = []

                for query in queries:
                    retrieved_candidates = results.get(query, [])
                    relevant_candidates = positive_map.get(query, set())

                    true_positives = sum(
                        1 for c in retrieved_candidates
                        if c in relevant_candidates
                    )

                    if query in positive_map and relevant_candidates:
                        if true_positives > 0:
                            positive_hits += 1

                        precision = true_positives / len(retrieved_candidates) \
                            if retrieved_candidates else 0.0
                        precision_list.append(precision)

                        recall = true_positives / len(relevant_candidates) \
                            if relevant_candidates else 0.0

                        if precision + recall > 0:
                            f1 = 2 * precision * recall / (precision + recall)
                        else:
                            f1 = 0.0
                        f1_list.append(f1)

                recall_at_k = (
                    positive_hits / total_positive if total_positive > 0 else None
                )
                avg_precision = (
                    np.mean(precision_list) if precision_list else None
                )
                avg_f1 = np.mean(f1_list) if f1_list else None

                negative_queries = [
                    q for q in queries
                    if q not in positive_map or not positive_map[q]
                ]
                negative_false_positives = sum(
                    1 for q in negative_queries if len(results.get(q, [])) > 0
                )
                total_negative = len(negative_queries)
                false_positive_rate = (
                    negative_false_positives / total_negative
                    if total_negative > 0 else None
                )

                config_results.append({
                    "k": k,
                    "threshold": threshold,
                    "recall@k": round(recall_at_k, 4)
                        if recall_at_k is not None else None,
                    "false_positive_rate": round(false_positive_rate, 4)
                        if false_positive_rate is not None else None,
                    "precision@k": round(avg_precision, 4)
                        if avg_precision is not None else None,
                    "f1@k": round(avg_f1, 4) if avg_f1 is not None else None
                })

            tagged_results.append({
                "dataset": sample_name,
                "model": model_name,
                "results": config_results
            })

        self._save_results(output_file, tagged_results)


    def benchmark_reranker(self, model_name, task_name, output_file, label_map=None):

        tasks = self._load_tasks(task_name)
        dataframes = self._load_data(tasks, label_map)

        model_path = os.path.join(self.model_folder, model_name)

        for df, sample_name in zip(dataframes, tasks):
            reranker = Reranker(model_path, df)
            logits = reranker.get_logits()
            results = []

            for threshold in RERANKING_CONFIGS:
                preds = reranker.get_prediction(logits, threshold)
                report_dict = classification_report(
                    df['label'], preds, output_dict=True
                )
                tagged_results = {
                    "dataset": task_name,
                    "model": model_name,
                    "threshold": threshold,
                    "classification_report": report_dict
                }
                self._save_results(output_file, [tagged_results])



    def benchmark_pipeline(
            self, retriever_name, reranker_p_name, reranker_s_name,
            task_name, output_file
        ):

        tasks = self._load_tasks(task_name)
        dataframes = self._load_data(tasks)

        retriever_model_path = os.path.join(self.model_folder, retriever_name)
        reranker_p_path = os.path.join(self.model_folder, reranker_p_name)
        reranker_s_path = os.path.join(self.model_folder, reranker_s_name)

        all_results = []

        for df, sample_name in zip(dataframes, tasks):
            queries = list(df['sentence1'].unique())
            candidates = df['sentence2'].tolist()

            true_labels_map = {}
            for _, row in df.iterrows():
                true_labels_map[(row['sentence1'], row['sentence2'])] = LABELS[row['label']]

            for config in tqdm(PIPELINE_CONFIGS, desc="Pipeline configurations"):
                k = config["k"]
                quote_threshold = config["quote_threshold"]
                retrieval_threshold = config["retrieval_threshold"]
                reranker_p_threshold = config["reranking_p_threshold"] 
                reranker_s_threshold = config["reranking_s_threshold"]

                pred_labels = run_pipeline(
                    retriever_model_path,
                    reranker_p_path,
                    reranker_s_path,
                    queries,
                    candidates,
                    k,
                    quote_threshold,
                    retrieval_threshold,
                    reranker_p_threshold,
                    reranker_s_threshold
                )

                y_true = [] 
                y_pred = []

                for pair in pred_labels:
                    true_label = true_labels_map.get(pair, "irrelevant")
                    pred_label = pred_labels[pair]

                    y_true.append(true_label)
                    y_pred.append(pred_label)

                report_dict = classification_report(
                    y_true, y_pred,
                    labels=["quote", "fuzzy_quote", "paraphrase", "similar_sentence", "irrelevant"],
                    output_dict=True,
                    zero_division=0
                )

                conf_matrix = confusion_matrix(y_true, y_pred, labels=["quote", "fuzzy_quote", "paraphrase", "similar_sentence", "irrelevant"])

                y_true_binary = [int(lbl != "irrelevant") for lbl in y_true]
                y_pred_binary = [int(lbl != "irrelevant") for lbl in y_pred]

                binary_report = classification_report(
                    y_true_binary, y_pred_binary,
                    target_names=["irrelevant", "relevant"],
                    output_dict=True,
                    zero_division=0
                )

                result_entry = {
                    "dataset": sample_name,
                    "retriever": retriever_name,
                    "reranker_p": reranker_p_name,
                    "reranker_s": reranker_s_name,
                    "k": k,
                    "retrieval_threshold": retrieval_threshold,
                    "reranker_p_threshold": reranker_p_threshold,
                    "reranker_s_threshold": reranker_s_threshold,
                    "classification_report": report_dict,
                    "binary_classification_report": binary_report,
                    "confusion_matrix": {
                        "labels": ["quote", "fuzzy_quote", "paraphrase", "similar_sentence", "irrelevant"],
                        "matrix": conf_matrix.tolist()
                    }
                }

                all_results.append(result_entry)

        self._save_results(output_file, all_results)

    def _load_data(self, tasks, label_map=None):
        dataframes = []
        for name in tasks:
            file_path = os.path.join(self.task_folder, name)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            if label_map:
                 df['label'] = df['label'].map(
                lambda x: TASK_LABEL_MAP.get(label_map).get(str(x))
            )
            else:
                df['label'] = df['label'].map(
                    lambda x: TASK_LABEL_MAP.get(name[:2]).get(str(x))
                )
            dataframes.append(df)
        return dataframes


    def _load_tasks(self, task_name):
        pattern = re.compile(fr'^{re.escape(task_name)}[SM]_\d+\.json$')
        return [
            os.path.basename(f)
            for f in os.listdir(self.task_folder)
            if os.path.isfile(os.path.join(self.task_folder, f)) and pattern.match(f)
        ]


    def _save_results(self, output_file, new_entries):
        output_path = os.path.join(self.result_folder, output_file)
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                try:
                    existing = json.load(f)
                    if not isinstance(existing, list):
                        existing = [existing]
                except json.JSONDecodeError:
                    existing = []
        else:
            existing = []

        existing.extend(new_entries)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)


