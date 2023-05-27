if [ ! -d "./data/raw" ]; then
  mkdir -p ./data/raw
fi
kaggle competitions download -c vesuvius-challenge-ink-detection -p ./data/raw