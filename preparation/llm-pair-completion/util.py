from typing import List
import random
import math

import pandas as pd
import spacy 

nlp = spacy.load('la_core_web_lg')

def read_sentences(file_path: str) -> List[str]:
    df = pd.read_csv(file_path)
    return df['sentence'].str.strip().str.strip('"').tolist()

def extract_substructures(sentence: str, min_len_subtree: int) -> List[str]:
    doc = nlp(sentence)
    substructures = set()
    len_sentence = len(sentence.split(' '))
    for token in doc:
        subtree = " ".join([t.text for t in token.subtree])
        subtree = subtree.replace(',', '').strip()
        subtree = subtree.replace('  ', ' ')
        len_subtree = len(subtree.split(' '))
        if(len_subtree > min_len_subtree and len_subtree < len_sentence):
            substructures.add(subtree)
    return list(substructures)


def choose_substructures(sentence: str) -> List[str]:

    len_sentence = len(sentence.split(' '))
    min_len_subtree = (
        math.ceil(len_sentence / 3) if len_sentence < 25 
        else math.ceil(len_sentence / 7) if len_sentence < 50
        else math.ceil(len_sentence / 15)
    )

    substructures = extract_substructures(sentence, min_len_subtree)
    chosen_substructures = []
    num_subtrees = math.ceil(len(substructures) / 3)
    selection = random.sample(substructures, num_subtrees)

    # Remove subsets from the sampled selection
    selection.sort(key=len, reverse=True)  # Sort longest first
    unique_selection = []

    for sub in selection:
        sub_set = set(sub.split())  # Convert substructure to a set of words
        if not any(sub_set.issubset(set(added.split())) for added in unique_selection):
            unique_selection.append(sub)

    return unique_selection




if __name__ == "__main__":
    substructures = choose_substructures("Ecce audiant hoc illi, qui maxime ecclesiarum localium, id est coenobiorum, archimandritis detrahunt, quoties gregis sui patiuntur detrimentum, et cum ipsi uacent otio, temere operarios Dei diiudicant, ubi aliquos ex eis, qui hortatu ipsorum conuersi sunt ad saeculum relabi conspiciunt.")
    for struct in substructures:
        print(struct)
