#!/bin/bash

NUM_REQUESTS=20
SAMPLE_SIZE=2500
BATCH_SIZE=50

INPUT="corpus.jsonl"

RESULT_DIR="results"

mkdir -p "$RESULT_DIR"

###########################
# Paraphrases 
###########################

parallel --jobs "$NUM_REQUESTS" \
  "python3 paraphrases.py $INPUT $RESULT_DIR/paraphrases_{#}.json $SAMPLE_SIZE $BATCH_SIZE" \
  ::: $(seq 1 $NUM_REQUESTS)

jq -s '.' "$RESULT_DIR"/paraphrases_*.json > "$RESULT_DIR"/paraphrases.json 
jq 'flatten(1)' "$RESULT_DIR"/paraphrases.json > tmp.json && mv tmp.json "$RESULT_DIR"/paraphrases.json

rm "$RESULT_DIR"/paraphrases_*.json

echo "Paraphrases finished."


