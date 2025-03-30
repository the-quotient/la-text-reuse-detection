import os
import json
import glob
from retrieval import Retrieval
from reranking import Reranking

def load_jsonl(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return [json.loads(line)["sentence"] for line in f]

def run_pipeline(
    query_file, candidate_folder, output_folder,
    retriever_path, reranker_p_path, reranker_s_path,
    fuzzy_threshold, k, retrieve_threshold,
    rerank_threshold_p, rerank_threshold_s
):

    os.makedirs(output_folder, exist_ok=True)

    # Step 1: Load and embed queries once
    queries = load_jsonl(query_file)
    retriever = Retrieval(retriever_path, queries)

    # Step 2: Process each candidate file
    for cand_path in glob.glob(os.path.join(candidate_folder, '*.jsonl')):
        candidates = load_jsonl(cand_path)
        cand_filename = os.path.basename(cand_path)
        name_part = os.path.splitext(os.path.basename(query_file))[0]
        output_filename = 'Comp_' + name_part + '_' + os.path.splitext(cand_filename)[0] + '.json'
        output_path = os.path.join(output_folder, output_filename)

        print(f"\n Processing: {cand_filename}")
        retriever.embed_candidates(candidates)

        fuzzy_quotes, to_rerank = retriever.retrieve_dual(
            k=k,
            fuzzy_threshold=fuzzy_threshold,
            retrieve_threshold=retrieve_threshold
        )
        print(len(fuzzy_quotes), len(to_rerank))

        results = {
            "fuzzy_quotes": [],
            "paraphrases": [],
            "similar_sentences": []
        }

        results["fuzzy_quotes"].extend([
            {"query": q, "candidate": c} for q, c in fuzzy_quotes
        ])

        if to_rerank:
            reranker1 = Reranking(reranker_p_path)
            paraphrases = reranker1.predict(to_rerank, rerank_threshold_p)
            print(len(paraphrases))
            results["paraphrases"].extend([
                {"query": q, "candidate": c} for q, c in paraphrases
            ])

            remaining = [pair for pair in to_rerank if pair not in set(paraphrases)]
            if remaining:
                reranker2 = Reranking(reranker_s_path)
                similar_sentences = reranker2.predict(remaining, rerank_threshold_s)
                print(len(similar_sentences))
                results["similar_sentences"].extend([
                    {"query": q, "candidate": c} for q, c in similar_sentences
                ])

        filtered = {k: v for k, v in results.items() if v}
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(filtered, f, indent=2, ensure_ascii=False)

        print(f" Saved: {output_path}")
