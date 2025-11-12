import os
from huggingface_hub import snapshot_download

HF_TOKEN = os.environ.get("HF_TOKEN")
model="meta-llama/Llama-3.1-8B-Instruct"

# Downloads the "original/*" files from the model repo into a local directory
snapshot_download(
    repo_id=model,
    allow_patterns=["original/*"],
    local_dir="Meta-Llama-3.1-8B-Instruct",
)