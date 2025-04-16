import os
import sys
import pandas as pd
from tqdm.notebook import tqdm

sys.path.append(os.path.abspath('.'))

from .fuzzy_span_aligner import fuzzy_match
from .retrieval import Retriever
from .reranking import Reranker


def run_pipeline(
    retriever_model_path,
    reranker_p_path,
    reranker_s_path,
    queries,
    candidates,
    k,
    quote_threshold,
    reranking_threshold,
    reranker_p_threshold,
    reranker_s_threshold,
    query_embeddings=None
): 
    retriever = Retriever(
        retriever_model_path, queries, 
        query_embeddings=query_embeddings, candidates=candidates
    )
    retrieved_quotes, to_process = retriever.retrieve_dual(
        k, quote_threshold, reranking_threshold
    )

    pred_labels = {}

    # Quotes
    for pair in retrieved_quotes:
        pred_labels[pair] = "quote"

    # Fuzzy quotes
    remaining_after_fuzzy = []
    for pair in tqdm(to_process):
        if fuzzy_match(pair[0], pair[1]):
            pred_labels[pair] = "fuzzy_quote"
        else:
            remaining_after_fuzzy.append(pair)

    # Paraphrases
    paraphrase_candidates = pd.DataFrame(
        remaining_after_fuzzy, columns=["sentence1", "sentence2"]
    )
    reranker_p = Reranker(reranker_p_path, paraphrase_candidates)
    logits_p = reranker_p.get_logits()
    preds_p = reranker_p.get_prediction(logits_p, reranker_p_threshold)

    remaining_after_p = []
    for idx, pred in enumerate(preds_p):
        pair = tuple(remaining_after_fuzzy[idx])
        if pred:
            pred_labels[pair] = "paraphrase"
        else:
            remaining_after_p.append(pair)

    # Similar sentences
    similar_candidates = pd.DataFrame(
        remaining_after_p, columns=["sentence1", "sentence2"]
    )
    reranker_s = Reranker(reranker_s_path, similar_candidates)
    logits_s = reranker_s.get_logits()
    preds_s = reranker_s.get_prediction(logits_s, reranker_s_threshold)

    for idx, pred in enumerate(preds_s):
        pair = tuple(remaining_after_p[idx])
        if pred:
            pred_labels[pair] = "similar_sentence"
        else:
            pred_labels[pair] = "irrelevant"

    return pred_labels


