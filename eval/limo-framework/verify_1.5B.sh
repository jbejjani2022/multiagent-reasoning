CUDA_VISIBLE_DEVICES='0' \
python verify.py \
--model_name_or_path "/n/netscratch/sham_lab/Everyone/jbejjani/output/deepseek_r1_1.5B_full_sft/maxcutoff_32768/32B_correct_verifications_of_1.5B_math500_solutions" \
--data_path "data/proposed_solutions/32B_generator/aime.jsonl" \
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

# --model_name_or_path "/n/netscratch/sham_lab/Everyone/jbejjani/output/deepseek_r1_1.5B_full_sft/maxcutoff_32768/math500_32B_correct_verifications_of_32B_solutions"
# --model_name_or_path "/n/holylabs/LABS/sham_lab/Users/jbejjani/DeepSeek-V3/models/DeepSeek-R1-Distill-Qwen-1.5B"
# --model_name_or_path "/n/holylabs/LABS/sham_lab/Users/jbejjani/multiagent-reasoning/train/LLaMA-Factory/output/math500_32B_correct_verifications/maxcutoff_4096/deepseek_r1_32B_lora_sft" \