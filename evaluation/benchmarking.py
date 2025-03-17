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

    
    def benchmark(self):

        not_none_models = [
            (i, model)
            for i, model in enumerate(self.models)
            if model is not None
        ]
        if len(not_none_models) > 1 and len(not_none_models) < 4:
            print(" Error! Wrong number of models selected!")
            return

        if len(not_none_models) == 4:
            print("Starting Benchmarking in Pipeline Mode.")
            return self._benchmark_pipeline()

        print("Starting Benchmarking in Single Model Mode.")

        index, model = not_none_models[0]
        if index == 0:
            print("Starting Retriever Benchmarking.")
            self._benchmark_retriever()
        else:
            print("Starting Reranker Benchmarking.")
            data = self._load_data(self.samples.get(['F', 'P', 'C'][index - 1]))
            self._benchmark_reranker(os.path.join(self.model_folder, model), data)

    
    def _load_data(self, sample_names):
        dataframes = []
        expected_cols = {'sentence1', 'sentence2', 'label'}
        for name in sample_names:
            file_path = os.path.join(self.data_folder, f"{name}.csv")
            df = pd.read_csv(file_path)
            if not expected_cols.issubset(df.columns):
                raise ValueError(
                    f"CSV file '{file_path}' must contain columns: {expected_cols}"
                )
            df['label'] = df['label'].apply(
                lambda x: 0 if str(x).lower() == "irrelevant" else 1
            )
            dataframes.append(df)
        if len(dataframes) > 1:
            data = pd.concat(dataframes, ignore_index=True)
        else:
            data = dataframes[0]
        return data

    
    def _benchmark_retriever(self):
        
        data = self._load_data(self.samples.get('G'))
        
        positive_map = defaultdict(set)
        for _, row in data.iterrows():
            if row['label'] == 1:
                positive_map[row['sentence1']].add(row['sentence2'])
                
        queries = list(data['sentence1'].unique())
        candidates = data['sentence2'].tolist()

        retrieval = Retrieval(os.path.join(self.model_folder, self.models[0]), queries, candidates)

        reports = []
        print("Results:  ") 
        for config in self.retrieval_configs:
            k = config.get("k")
            threshold = config.get("threshold")
            results = retrieval.retrieve(k, threshold)

            positive_hits = 0
            total_positive = len([q for q in queries if q in positive_map])
            for query in queries:
                retrieved_candidates = results.get(query, [])
                if query in positive_map:
                    if any(candidate in positive_map[query]
                           for candidate in retrieved_candidates):
                        positive_hits += 1
            recall_at_k = positive_hits / total_positive if total_positive > 0 else None

            negative_queries = [
                q for q in queries
                if q not in positive_map or not positive_map[q]
            ]
            negative_false_positives = sum(
                1 for q in negative_queries if len(results.get(q, [])) > 0
            )
            total_negative = len(negative_queries)
            false_positive_rate = (negative_false_positives / total_negative
                                   if total_negative > 0 else None)
            
            print(f"Retrieval (k={k}, threshold={threshold}): "
                  f"Recall@{k}: {recall_at_k:.4f}, FPR: {false_positive_rate:.4f}")

            # TODO: Implement file logging 

    
    def _benchmark_reranker(self, reranker, data):
        reranking = Reranking(reranker, data)
        reports = []
        for config in self.reranker_configs:
            threshold = config.get("threshold")
            preds = reranking.predict(threshold)
            report_str = classification_report(data['label'], preds)
            print(f"Reranker (threshold={threshold}):\n{report_str}")
            
        # TODO: Implement file logging 

    
    def _benchmark_pipeline(self):

        # TODO: Implement logic 
        
        reports = []
        for config in self.pipeline_configs:
            threshold = config.get("threshold")
            dummy_pipeline_score = threshold
            print(f"Pipeline (threshold={threshold}): "
                  f"Score: {dummy_pipeline_score:.4f}")

        # TODO: Implement file logging 
