#!/bin/bash

NUM_REQUESTS=20
SAMPLE_SIZE=850
BATCH_SIZE=50

INPUT="corpus.jsonl"
RESULT_DIR="results"

mkdir -p "$RESULT_DIR"

###########################
# Fuzzy Quotes
###########################

parallel --jobs "$NUM_REQUESTS" \
  "python3 fuzzy_quotes.py $INPUT $RESULT_DIR/fuzzy_quotes_{#}.json $SAMPLE_SIZE $BATCH_SIZE" \
  ::: $(seq 1 $NUM_REQUESTS)

jq -s '.' "$RESULT_DIR"/fuzzy_quotes_*.json > "$RESULT_DIR"/fuzzy_quotes.json 
jq 'flatten(1)' "$RESULT_DIR"/fuzzy_quotes.json > tmp.json && mv tmp.json "$RESULT_DIR"/fuzzy_quotes.json

rm "$RESULT_DIR"/fuzzy_quotes_*.json

echo "Fuzzy Quotes finished."


