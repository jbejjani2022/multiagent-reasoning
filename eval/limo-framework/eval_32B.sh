CUDA_VISIBLE_DEVICES='0,1' \
python eval.py \
--model_name_or_path "/n/holylabs/LABS/sham_lab/Users/jbejjani/multiagent-reasoning/train/LLaMA-Factory/output/math500_32B_correct_solutions/maxcutoff_4096/deepseek_r1_32B_lora_sft" \
--data_name "math" \
--prompt_type "qwen-instruct" \
--temperature 0.6 \
--start_idx 0 \
--end_idx -1 \
--n_sampling 1 \
--k 1 \
--split "test" \
--max_tokens 32768 \
--seed 0 \
--top_p 1 \
--surround_with_messages \

# --model_name_or_path "/n/holylabs/LABS/sham_lab/Users/jbejjani/DeepSeek-V3/models/DeepSeek-R1-Distill-Qwen-32B"
# --model_name_or_path "/n/holylabs/LABS/sham_lab/Users/jbejjani/multiagent-reasoning/train/LLaMA-Factory/output/math500_32B_correct_solutions/maxcutoff_4096/deepseek_r1_32B_lora_sft" \
# --model_name_or_path "/n/holylabs/LABS/sham_lab/Users/jbejjani/DeepSeek-V3/models/DeepSeek-R1-Distill-Qwen-32B" \
# --max_tokens 15360