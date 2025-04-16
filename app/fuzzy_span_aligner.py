from rapidfuzz import fuzz
import spacy

nlp = spacy.load("la_core_web_lg")

def get_dynamic_window_bounds(len1, len2, min_limit=10, max_limit=20):
    avg_len = (len1 + len2) / 2
    min_window = max(min_limit, int(avg_len * 0.2))
    max_window = min(max_limit, int(avg_len * 0.6))
    return min_window, max_window


def fuzzy_match_spans_dynamic(s1, s2, threshold=80):
    doc1 = nlp(s1)
    doc2 = nlp(s2)

    tokens1 = [token.lemma_ for token in doc1 if not token.is_punct]
    tokens2 = [token.lemma_ for token in doc2 if not token.is_punct]

    min_window, max_window = get_dynamic_window_bounds(len(tokens1), len(tokens2))

    results = []

    for window_size in range(min_window, max_window + 1):
        for i in range(len(tokens1) - window_size + 1):
            span1 = tokens1[i:i+window_size]
            str1 = " ".join(span1)

            for j in range(len(tokens2) - window_size + 1):
                span2 = tokens2[j:j+window_size]
                str2 = " ".join(span2)

                score = fuzz.token_set_ratio(str1, str2)
                adjusted_score = score * (window_size / max_window)

                if adjusted_score >= threshold:
                    start1 = doc1[i].idx
                    end1 = doc1[i + window_size - 1].idx + \
                        len(doc1[i + window_size - 1])
                    start2 = doc2[j].idx
                    end2 = doc2[j + window_size - 1].idx + \
                        len(doc2[j + window_size - 1])

                    results.append({
                        "score": score,
                        "s1_tokens": span1,
                        "s2_tokens": span2,
                        "s1_text": s1[start1:end1],
                        "s2_text": s2[start2:end2],
                        "s1_span": (start1, end1),
                        "s2_span": (start2, end2)
                    })

    return sorted(results, key=lambda x: x["score"], reverse=True)

def fuzzy_match(s1, s2):
    matches = fuzzy_match_spans_dynamic(s1, s2, threshold=80)
    return 1 if matches else 0



