##!/bin/bash

NUM_REQUESTS=20
SAMPLE_SIZE=2500
BATCH_SIZE=50

INPUT="corpus.jsonl"

RESULT_DIR="results"

mkdir -p "$RESULT_DIR"

##########################
# Similar sentences 
###########################

parallel --jobs "$NUM_REQUESTS" \
  "python3 similar_sentences.py $INPUT $RESULT_DIR/similar_sentences_{#}.json $SAMPLE_SIZE $BATCH_SIZE" \
  ::: $(seq 1 $NUM_REQUESTS)

jq -s '.' "$RESULT_DIR"/similar_sentences_*.json > "$RESULT_DIR"/similar_sentences.json
jq 'flatten(1)' "$RESULT_DIR"/similar_sentences.json > tmp.json && mv tmp.json "$RESULT_DIR"/similar_sentences.json

rm "$RESULT_DIR"/similar_sentences_*.json

echo "Similar sentences finished."
