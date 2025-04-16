import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder
from tqdm.notebook import tqdm


class Reranker:

    def __init__(self, model_name, data=None):
        self.reranker = CrossEncoder(
            model_name,
            max_length=256
        )
        self.data = data


    def get_logits(self):
        sents1 = self.data['sentence1']
        sents2 = self.data['sentence2']
        result= []
        for s1, s2 in tqdm(zip(sents1, sents2), total=len(sents1)):
            logits = self.reranker.predict([(s1, s2)])
            result.append(logits)
        return result


    def get_prediction(self, logits, threshold):
        return (np.array(logits).flatten() > threshold).astype(int).tolist()


    def predict(self, pairs, threshold):
        df = pd.DataFrame(pairs, columns=["sentence1", "sentence2"])
        self.data = df
        logits = self.get_logits()
        preds = self.get_prediction(logits, threshold)
        return [pair for pair, pred in zip(pairs, preds) if pred == 1]
