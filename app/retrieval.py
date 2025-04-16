from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm.notebook import tqdm


class Retriever:

    def __init__(self, model_name, queries, query_embeddings=None, candidates=None):
        self.model = SentenceTransformer(model_name)
        self.query_sentences = queries
        self.query_embeddings = None
        self.candidate_sentences = None
        self.candidate_embeddings = None
        if query_embeddings is None:
            print("\nEmbedding queries...")
            self.query_embeddings = self.model.encode(
                queries, show_progress_bar=True
            )
        else:
            self.query_embeddings = query_embeddings
        if candidates is not None:
            print("\nEmbedding candidates...")
            self.candidate_sentences = candidates
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


    def retrieve(self, k, threshold):
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


    def retrieve_dual(self, k, quote_threshold, reranking_threshold):

        quotes = []
        to_rerank = []

        for query, q_emb in tqdm(zip(self.query_sentences, self.query_embeddings), total=len(self.query_sentences)):
            sims = cosine_similarity([q_emb], self.candidate_embeddings)[0]
            sorted_indices = np.argsort(-sims)[:k]

            for i in sorted_indices:
                sim_score = sims[i]
                candidate = self.candidate_sentences[i]

                if sim_score >= quote_threshold:
                    quotes.append((query, candidate))
                elif sim_score >= reranking_threshold:
                    to_rerank.append((query, candidate))

        return quotes, to_rerank

