
from huggingface_hub import snapshot_download
import subprocess

# export HF_ENDPOINT=https://hf-mirror.com
# subprocess.check_output(["export", "HF_ENDPOINT=https://hf-mirror.com"])
# repo_id = 'meta-llama/Llama-3.2-8B-Instruct'
repo_id = 'meta-llama/Meta-Llama-3-8B-Instruct'
# microsoft/Phi-3-medium-4k-instruct
# lmsys/vicuna-7b-v1.5
# repo_id = "meta-llama/Llama-3.2-3B-Instruct"

local_dir = "/your/path/to/checkpoints/" + repo_id.split("/")[1]
snapshot_download(repo_id=repo_id, local_dir=local_dir)
