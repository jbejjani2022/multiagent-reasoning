"""
    https://docs.vllm.ai/en/latest/getting_started/quickstart.html#quickstart-online
    
    Request interactive session with 2 GPUs:
    salloc --partition=gpu_requeue --ntasks=1 --cpus-per-task=16 --mem=64G --gres=gpu:2 --time=00-03:00:00
    
    Serve model with:
    vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --tensor-parallel-size 2 --max-model-len 32768 --enforce-eager --download-dir /n/holylabs/LABS/sham_lab/Users/jbejjani/DeepSeek-V3/models
    
    If --model-dir is not specified, default is ~/.cache/huggingface (but note that my home dir only has 100GB)
    
    Loading the weights for deepseek-ai/DeepSeek-R1-Distill-Qwen-32B takes ~30.7GB
    
    List models being served:
    curl http://localhost:8000/v1/models

    Run inference for a prompt:
    curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "prompt": "What is the capital of France?",
        "max_tokens": 50,
        "temperature": 0.6
    }'
"""

import requests


url = "http://localhost:8000/v1/completions"
data = {
    "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "prompt": "How many occurrences of 'r' does the word 'strawberry' have? <think>",
    "max_tokens": 500,
    "temperature": 0.6
}

response = requests.post(url, json=data)
print(response.json())
