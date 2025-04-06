#!/bin/bash

NUM_REQUESTS=15
SAMPLE_SIZE=1000
BATCH_SIZE=100

INPUT="corpus.jsonl"
RESULT_DIR="results4"

mkdir -p "$RESULT_DIR"

###########################
# Fuzzy Quotes
###########################

parallel --jobs "$NUM_REQUESTS" \
  "python3 fuzzy_quotes_2.py $INPUT $RESULT_DIR/fuzzy_quotes_2_{#}.json $SAMPLE_SIZE $BATCH_SIZE" \
  ::: $(seq 1 $NUM_REQUESTS)

jq -s '.' "$RESULT_DIR"/fuzzy_quotes_2_*.json > "$RESULT_DIR"/fuzzy_quotes_2.json
jq 'flatten(1)' "$RESULT_DIR"/fuzzy_quotes_2.json > tmp.json && mv tmp.json "$RESULT_DIR"/fuzzy_quotes_2.json

rm "$RESULT_DIR"/fuzzy_quotes_2_*.json

echo "Fuzzy Quotes finished."


