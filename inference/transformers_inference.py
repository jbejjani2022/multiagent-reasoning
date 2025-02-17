import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Device: {device}")

input = "What is the capital of France? <think>\n"
input_tokens = tokenizer(input, return_tensors="pt")
input_tokens = {key: value.to(device) for key, value in input_tokens.items()}

with torch.no_grad():
    output = model.generate(**input_tokens, max_length=50, temperature=0.6)
    
output_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Output\n********\n", output_text)
