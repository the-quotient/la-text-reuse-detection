import numpy as np
from sentence_transformers import CrossEncoder
from tqdm import tqdm

class Reranking:

    def __init__(self, model_name, data):
        self.reranker = CrossEncoder(model_name)
        self.data = data

    def predict(self):
        preds = []
        sents1 = self.data['sentence1']
        sents2 = self.data['sentence2']
        for s1, s2 in tqdm(zip(sents1, sents2), total=len(sents1)):
            logits = self.reranker.predict([(s1, s2)])
            pred = int(np.argmax(logits, axis=1)[0])
            preds.append(pred)
        return preds

