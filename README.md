# Setup

Go to Sagemaker AI service -> Admin configuration -> Domains
Create User and Launch, jupyter with size 100GB max

## Service quota

Run from local to increase quota for ml.g5.48x large
```bash
./sagemaker/service-quota.sh
```

## Run Sagemaker

Git clone inside jupyter
```bash
./sagemaker/conda-setup.sh
```

Run 01 for data_prep and model training
Run 02 deploy using sagemaker and run evaluation
Run 03 for model deployment using bedrock custom import and run evaluation

See see launched jobs: `Go to Jobs -> training -> instances and logs will be visible here too`

## Resources Referred
Refer `resources.md`

## Notes
- `jumpstart/6-evaluate.py` parallelizes per-example metric computation across available CPU cores to speed up large evaluations.
- Toggle the `INCLUDE_BERT_SCORE` variable in `jumpstart/6-evaluate.py` to enable or disable BERTScore F1 during evaluation runs.
- Evaluation artifacts now include `per_example_metrics.csv` plus a combined `metric_boxplots.png` with BLEU-4 and Levenshtein distributions.
