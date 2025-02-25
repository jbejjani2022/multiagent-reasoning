"""
    Request an interactive session with one A100 80GB GPU:
    salloc --partition=gpu_requeue --ntasks=1 --cpus-per-task=16 --mem=64G --gres=gpu:nvidia_a100-sxm4-80gb:1 --time=00-01:00:00
    
    salloc --partition=kempner_requeue --ntasks=1 --cpus-per-task=16 --mem=64G --gres=gpu:nvidia_a100-sxm4-80gb:1 --time=00-01:00:00
    salloc --partition=kempner_requeue --account=kempner_sham_lab --ntasks=1 --cpus-per-task=16 --mem=64G --gres=gpu:nvidia_h100_80gb_hbm3:1 --time=00-01:00:00
    
    Or two A100 40GB GPUs:
    salloc --partition=gpu_requeue --ntasks=1 --cpus-per-task=16 --mem=64G --gres=gpu:nvidia_a100-sxm4-40gb:2 --time=00-02:00:00
    salloc --partition=kempner_requeue --account=kempner_sham_lab --ntasks=1 --cpus-per-task=16 --mem=64G --gres=gpu:2 --constraint=h100 --time=00-02:00:00

    Serve a model with vllm:
    vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --tensor-parallel-size 1 --max-model-len 15360 --enforce-eager --download-dir /n/holylabs/LABS/sham_lab/Users/jbejjani/DeepSeek-V3/models
    
    vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --tensor-parallel-size 2 --max-model-len 32768 --enforce-eager --download-dir /n/holylabs/LABS/sham_lab/Users/jbejjani/DeepSeek-V3/models
    
    Then set:
    base_url="http://127.0.0.1:8000/v1"
    
    Loading the weights for deepseek-ai/DeepSeek-R1-Distill-Qwen-32B takes 61.0608 GB
    Can split this across multiple devices with --tensor-parallel-size > 1
    
    Or serve with sglang:
    
    python -m sglang.launch_server --model /n/holylabs/LABS/sham_lab/Users/jbejjani/DeepSeek-V3/models/DeepSeek-R1-Distill-Qwen-1.5B --trust-remote-code --tp 1
    ^takes like ~3GB
    
    python -m sglang.launch_server --model /n/holylabs/LABS/sham_lab/Users/jbejjani/DeepSeek-V3/models/DeepSeek-R1-Distill-Qwen-32B --trust-remote-code --tp 2
    ^takes up ~39.68GB per device
    
    And set:
    base_url="http://127.0.0.1:30000/v1"
"""

import openai


base_url = "http://127.0.0.1:30000/v1"
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

client = openai.Client(base_url=base_url, api_key="None")

response = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    max_tokens=64,
    temperature=0.6
)
print(response)
