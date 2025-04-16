import os
import json
import glob

from .pipeline import run_pipeline


def load_jsonl(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return [json.loads(line)["sentence"] for line in f]


def detect(
    retriever_model_path,
    reranker_p_path,
    reranker_s_path,
    k,
    quote_threshold,
    reranking_threshold,
    reranker_p_threshold,
    reranker_s_threshold,
    queries=None,
    candidates=None,
    query_file=None,
    candidate_folder=None,
    output_folder=None,
    output_file=None,
):

    if query_file and candidate_folder:
        queries = load_jsonl(query_file)
        for cand_path in glob.glob(os.path.join(candidate_folder, '*.jsonl')):
            candidates = load_jsonl(cand_path)
            cand_filename = os.path.basename(cand_path)
            name_part = os.path.splitext(os.path.basename(query_file))[0]
            output_filename = (
                    'Comp_' + name_part + '_' + 
                    os.path.splitext(cand_filename)[0] + '.json'
            )
            output_path = os.path.join(output_folder, output_filename)
            print(f"\n Processing: {cand_filename}")
            pred_labels = run_pipeline(
                retriever_model_path,
                reranker_p_path,
                reranker_s_path,
                queries,
                candidates,
                k,
                quote_threshold,
                reranking_threshold,
                reranker_p_threshold,
                reranker_s_threshold
            )

            categorized_results = {
                "quote": [],
                "fuzzy_quote": [],
                "paraphrase": [],
                "similar_sentence": [],
                "irrelevant": []
            }

            for pair, label in pred_labels.items():
                categorized_results[label].append(list(pair))

            print(" Category counts:")
            for category, pairs in categorized_results.items():
                print(f"  {category}: {len(pairs)} pairs")


            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(categorized_results, f, indent=2, ensure_ascii=False)
            print(f" Saved: {output_path}")

    else:
        pred_labels = run_pipeline(
            retriever_model_path,
            reranker_p_path,
            reranker_s_path,
            queries,
            candidates,
            k,
            quote_threshold,
            reranking_threshold,
            reranker_p_threshold,
            reranker_s_threshold
        )

        categorized_results = {
            "quote": [],
            "fuzzy_quote": [],
            "paraphrase": [],
            "similar_sentence": [],
            "irrelevant": []
        }

        for pair, label in pred_labels.items():
            categorized_results[label].append(list(pair))

        print(" Category counts:")
        for category, pairs in categorized_results.items():
            print(f"  {category}: {len(pairs)} pairs")

        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(categorized_results, f, indent=2, ensure_ascii=False)
            print(f" Saved: {output_file}")

        return categorized_results


