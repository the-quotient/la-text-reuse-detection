import json
import os
import sys
import warnings
from collections import defaultdict
from itertools import product

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer, CrossEncoder

sys.path.append(os.path.abspath('..'))
from app import Retrieval, Reranking


class BenchmarkRunner:

    def __init__(self, model_folder, models, data_folder, samples, result_folder,
                 pipeline_configs, retrieval_configs, reranker_configs):
        self.model_folder = model_folder
        self.models = models
        self.data_folder = data_folder
        self.samples = samples
        self.result_folder = result_folder
        self.pipeline_configs = pipeline_configs
        self.retrieval_configs = retrieval_configs
        self.reranker_configs = reranker_configs

    def benchmark(self, output_file):
        not_none_models = [
            (i, model)
            for i, model in enumerate(self.models)
            if model is not None
        ]

        if len(not_none_models) > 1 and len(not_none_models) < 4:
            print("Error! Wrong number of models selected!")
            return

        if len(not_none_models) == 4:
            print("Starting Benchmarking in Pipeline Mode.")
            return self._benchmark_pipeline()

        print("Starting Benchmarking in Single Model Mode.")

        index, model = not_none_models[0]
        if index == 0:
            print("Starting Retriever Benchmarking.")
            self._benchmark_retriever(output_file)
        else:
            print("Starting Reranker Benchmarking.")
            sample_key = 'P' if index == 1 else 'S'
            data = self._load(self.samples.get(sample_key))
            for df, sample_name in zip(data, self.samples.get(sample_key)):
                self._benchmark_reranker(
                    os.path.join(self.model_folder, model),
                    df,
                    sample_name,
                    output_file
                )


    def _load(self, sample_names):
        dataframes = []
        for name in sample_names:
            file_path = os.path.join(self.data_folder, f"{name}.json")
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            df['label'] = df['label'].apply(
                lambda x: 0 if str(x).lower() == "irrelevant" else 1
            )
            dataframes.append(df)
        return dataframes



    def _benchmark_retriever(self, output_file):
        dataframes = self._load(self.samples.get('G'))
        model_name = self.models[0]
        model_path = os.path.join(self.model_folder, model_name)

        tagged_results = []

        for df, sample_name in zip(dataframes, self.samples.get('G')):
            positive_map = defaultdict(set)
            for _, row in df.iterrows():
                if row['label'] == 1:
                    positive_map[row['sentence1']].add(row['sentence2'])

            queries = list(df['sentence1'].unique())
            candidates = df['sentence2'].tolist()

            retrieval = Retrieval(model_path, queries, candidates)

            config_results = []
            for config in self.retrieval_configs:
                k = config.get("k")
                threshold = config.get("threshold")
                results = retrieval.retrieve(k, threshold)

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
        return tagged_results

    def _save_results(self, output_file, new_entries):
        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                try:
                    existing = json.load(f)
                    if not isinstance(existing, list):
                        existing = [existing]
                except json.JSONDecodeError:
                    existing = []
        else:
            existing = []

        existing.extend(new_entries)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)



    def _benchmark_reranker(self, reranker_path, data, sample_name, output_file):

        df = pd.DataFrame(data, columns=["sentence1", "sentence2"])
        reranking = Reranking(reranker_path, df)
        logits = reranking.get_logits()
        results = []

        for threshold in self.reranker_configs:
            preds = reranking.get_prediction(logits, threshold)
            report_dict = classification_report(data['label'], preds,
                                                output_dict=True)
            tagged_results = {
                "dataset": sample_name,
                "model": os.path.basename(reranker_path),
                "threshold": threshold,
                "classification_report": report_dict
            }
            self._save_results(output_file, [tagged_results])

        return results

    def _benchmark_pipeline(self):
        # TODO: Implement logic
        for config in self.pipeline_configs:
            threshold = config.get("threshold")
            dummy_pipeline_score = threshold
            print(f"Pipeline (threshold={threshold}): "
                  f"Score: {dummy_pipeline_score:.4f}")

        # TODO: Implement file logging

