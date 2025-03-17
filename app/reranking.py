from sentence_transformers import CrossEncoder
from tqdm import tqdm

class Reranking:
    
    def __init__(self, model_name, data):
        self.reranker = CrossEncoder(model_name)
        self.data = data

    def rerank(self, threshold=0.5):
        preds = self._compute_reranker_preds(threshold)
        positive_pairs = []
        for idx, pred in enumerate(preds):
            if pred == 1:
                pair = (self.data['sentence1'][idx],
                        self.data['sentence2'][idx])
                positive_pairs.append(pair)
        return positive_pairs

    def predict(self, threshold=0.5):
        return self._compute_reranker_preds(threshold)

    def _compute_reranker_preds(self, threshold):
        reranker_preds = []
        sents1 = self.data['sentence1']
        sents2 = self.data['sentence2']
        for s1, s2 in tqdm(zip(sents1, sents2), total=len(sents1)):
            score = self.reranker.predict([(s1, s2)])[0]
            pred_label = int(score > threshold)
            reranker_preds.append(pred_label)
        return reranker_preds
