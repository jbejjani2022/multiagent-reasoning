CUDA_VISIBLE_DEVICES='0,1' \
python verify.py \
--model_name_or_path "/n/holylabs/LABS/sham_lab/Users/jbejjani/DeepSeek-V3/models/DeepSeek-R1-Distill-Qwen-32B" \
--data_path "data/proposed_solutions/1.5B_generator/math.jsonl" \
--prompt_type "qwen-instruct" \
--temperature 0.6 \
--start_idx 0 \
--end_idx -1 \
--n_sampling 1 \
--k 1 \
--max_tokens 32768 \
--seed 0 \
--top_p 1 \
--surround_with_messages \

# --model_name_or_path "/n/holylabs/LABS/sham_lab/Users/jbejjani/DeepSeek-V3/models/DeepSeek-R1-Distill-Qwen-32B"