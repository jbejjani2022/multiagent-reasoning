"""
    https://docs.sglang.ai/start/send_request.html
    
    Request interactive session with 2 GPUs:
    salloc --partition=gpu_requeue --ntasks=1 --cpus-per-task=16 --mem=64G --gres=gpu:2 --time=00-03:00:00
    
    Serve model with:
    python -m sglang.launch_server --model /n/holylabs/LABS/sham_lab/Users/jbejjani/DeepSeek-V3/models/DeepSeek-R1-Distill-Qwen-1.5B --trust-remote-code --tp 2
    
    Ensure model has been downloaded, e.g.:
    huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --local-dir /n/holylabs/LABS/sham_lab/Users/jbejjani/DeepSeek-V3/models/DeepSeek-R1-Distill-Qwen-1.5B
    
    List models being served:
    curl http://localhost:30000/v1/models

    Run inference for a prompt:
    curl http://localhost:30000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "prompt": "What is the capital of France?",
        "max_tokens": 50,
        "temperature": 0.6
    }'
"""

import requests


url = "http://localhost:30000/v1/completions"
data = {
    "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "prompt": "How many occurrences of 'r' does the word 'strawberry' have? <think>",
    "max_tokens": 500,
    "temperature": 0.6
}

response = requests.post(url, json=data)
print(response.json())
