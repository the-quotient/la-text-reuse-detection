from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class Retrieval:

    def __init__(self, model_name, queries, candidates=None):
        print("Loading Sentence Transformer Model...")
        self.model = SentenceTransformer(model_name)
        self.query_sentences = queries
        self.candidate_sentences = candidates
        print("\nEmbedding queries...")
        self.query_embeddings = self.model.encode(
            queries, show_progress_bar=True
        )
        if candidates:
            print("\nEmbedding candidates...")
            self.candidate_embeddings = self.model.encode(
                candidates, show_progress_bar=True
            )
        print("\n")


    def embed_candidates(self, candidates):
        self.candidate_sentences = candidates
        print("Embedding candidates...")
        self.candidate_embeddings = self.model.encode(
            candidates, show_progress_bar=True
        )
        print(" Candidates embedded.\n")


    def retrieve(self, k=1, threshold=0.5):
        results = {}
        for query, q_emb in zip(self.query_sentences, self.query_embeddings):
            sims = cosine_similarity([q_emb], self.candidate_embeddings)[0]
            valid_indices = np.where(sims >= threshold)[0]
            if valid_indices.size > 0:
                sorted_indices = valid_indices[np.argsort(-sims[valid_indices])]
                top_k_indices = sorted_indices[:k]
                results[query] = [self.candidate_sentences[i] 
                                  for i in top_k_indices]
            else:
                results[query] = []
        return results


    def retrieve_dual(self, k, fuzzy_threshold, retrieve_threshold):

        fuzzy_quotes = []
        to_rerank = []

        for query, q_emb in zip(self.query_sentences, self.query_embeddings):
            sims = cosine_similarity([q_emb], self.candidate_embeddings)[0]
            sorted_indices = np.argsort(-sims)[:k]

            for i in sorted_indices:
                sim_score = sims[i]
                candidate = self.candidate_sentences[i]

                if sim_score >= fuzzy_threshold:
                    fuzzy_quotes.append((query, candidate))
                elif sim_score >= retrieve_threshold:
                    to_rerank.append((query, candidate))

        return fuzzy_quotes, to_rerank

