
# OLD

```bash
synthetic-data-kit -c configs/config.yaml ingest *.pdf
```
# Create

```bash
synthetic-data-kit -c configs/config.yaml create data/parsed/04adbf18-11c4-412a-922a-090e13ef595f.lance --type qa_generation -n 30
```

# Curate

```bash
synthetic-data-kit -c configs/config.yaml curate data/generated/04adbf18-11c4-412a-922a-090e13ef595f_qa_pairs.json -t 8.5
```

# Save

```bash
synthetic-data-kit save-as data/curated/04adbf18-11c4-412a-922a-090e13ef595f_qa_pairs_cleaned.json -f ft
```