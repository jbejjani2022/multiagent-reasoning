"""
    Request an interactive session with one A100 80GB GPU:
    salloc --partition=gpu_requeue --account=pslade_lab --ntasks=1 --cpus-per-task=16 --mem=128G --gres=gpu:nvidia_a100-sxm4-80gb:1 --time=00-02:00:00
    
    salloc --partition=kempner_requeue --ntasks=1 --cpus-per-task=16 --mem=64G --gres=gpu:nvidia_a100-sxm4-80gb:1 --time=00-01:00:00
    salloc --partition=kempner_requeue --account=kempner_sham_lab --ntasks=1 --cpus-per-task=16 --mem=128G --gres=gpu:nvidia_h100_80gb_hbm3:1 --time=00-02:00:00
    
    Or two A100 40GB GPUs:
    salloc --partition=gpu_requeue --account=pslade_lab --ntasks=1 --cpus-per-task=16 --mem=128G --gres=gpu:nvidia_a100-sxm4-80gb:2 --time=00-02:00:00
    
    salloc --partition=gpu_requeue --ntasks=1 --cpus-per-task=16 --mem=64G --gres=gpu:nvidia_a100-sxm4-40gb:2 --time=00-02:00:00
    salloc --partition=kempner_requeue --account=kempner_sham_lab --ntasks=1 --cpus-per-task=16 --mem=64G --gres=gpu:2 --constraint=h100 --time=00-02:00:00

    Serve a model with vllm:
    vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --tensor-parallel-size 1 --max-model-len 15360 --enforce-eager --download-dir /n/holylabs/LABS/sham_lab/Users/jbejjani/DeepSeek-V3/models
    vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --tensor-parallel-size 2 --max-model-len 32768 --enforce-eager --download-dir /n/holylabs/LABS/sham_lab/Users/jbejjani/DeepSeek-V3/models
    
    vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --tensor-parallel-size 2 --max-model-len 128000 --gpu-memory-utilization 0.96 --enforce-eager --download-dir /n/holylabs/LABS/sham_lab/Users/jbejjani/DeepSeek-V3/models
    
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
import json


base_url="http://127.0.0.1:30000/v1"
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

file_path = "aime_verification.jsonl"
client = openai.Client(base_url=base_url, api_key="None")

system_prompt = "You are a critical verifier tasked with evaluating mathematical problem-solving. You will be presented with a question and a proposed solution. Your job is to carefully go over and analyze the solution. Follow the instructions."
instructions = "INSTRUCTIONS: Go over each step in the proposed solution and check whether it is mathematically correct. Think out loud. If you reach a step that is incorrect, stop and reply 'FINAL VERIFICATION ANSWER: False'. If you get to the end of all the steps and each step was correct, reply 'FINAL VERIFICATION ANSWER: True'."
problems = {}

with open(file_path, "r", encoding="utf-8") as file:
    for line in file:
        obj = json.loads(line)  # Parse the JSON object
        question = obj.get("question", "")
        generated_responses = obj.get("generated_responses", "")
        if generated_responses:
            generated_response = generated_responses[0]
        is_correct = obj.get("is_correct", "")
        problems[f"QUESTION: {question}\nPROPOSED SOLUTION: {generated_response}\n"] = is_correct

with open('out2.txt', 'w') as f:
    for problem, is_correct in problems.items():
        c = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": problem + instructions}
            ],
            max_tokens=32768,
            temperature=0.6
        )
        out = c.choices[0].message.content
        print(out)
        f.write(out + "\n" + f"************************\nis_correct = {is_correct}\n************************\n")
