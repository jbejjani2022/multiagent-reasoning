"""
    Request an interactive session with one A100 80GB GPU:
    salloc --partition=gpu_requeue --ntasks=1 --cpus-per-task=16 --mem=64G --gres=gpu:nvidia_a100-sxm4-80gb:1 --time=00-01:00:00
"""

from sglang.utils import (
    execute_shell_command,
    wait_for_server,
    terminate_process,
    print_highlight,
)
import openai

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
model_path = "/n/holylabs/LABS/sham_lab/Users/jbejjani/DeepSeek-V3/models/DeepSeek-R1-Distill-Qwen-1.5B"

server_process = execute_shell_command(
    f"""
python -m sglang.launch_server --model {model_path} \
--port 30000 --host 0.0.0.0 --trust-remote-code --tp 1
"""
)

wait_for_server("http://localhost:30000")

client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")

response = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    max_tokens=64,
    temperature=0.6
)
print_highlight(response)
