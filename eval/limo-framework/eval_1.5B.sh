CUDA_VISIBLE_DEVICES='0,1' \
python eval.py \
--model_name_or_path '/n/netscratch/sham_lab/Everyone/jbejjani/verl/checkpoints/verl_ppo/1.5B_math500_ppo/global_step_620/actor/hf_model' \
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

# /n/netscratch/sham_lab/Everyone/jbejjani/output/deepseek_r1_1.5B_lora_kto/maxcutoff_16384/1.5B_math500_pass@5_solutions/3ep
# /n/netscratch/sham_lab/Everyone/jbejjani/output/deepseek_r1_1.5B_lora_kto/maxcutoff_16384/1.5B_math500_solutions/3ep
# /n/netscratch/sham_lab/Everyone/jbejjani/output/deepseek_r1_1.5B_lora_sft/maxcutoff_32768/1.5B_pass@5_correct_math500_solutions/3ep/new_hyperparams/cosine
# /n/netscratch/sham_lab/Everyone/jbejjani/output/deepseek_r1_1.5B_lora_sft/maxcutoff_32768/1.5B_pass@5_correct_math500_solutions/3ep/new_hyperparams
# --model_name_or_path "/n/netscratch/sham_lab/Everyone/jbejjani/output/deepseek_r1_1.5B_lora_sft/maxcutoff_32768/1.5B_pass@5_correct_math500_solutions/1ep"
# --model_name_or_path "/n/holylabs/LABS/sham_lab/Users/jbejjani/DeepSeek-V3/models/DeepSeek-R1-Distill-Qwen-1.5B"
# /n/holylabs/LABS/sham_lab/Users/jbejjani/multiagent-reasoning/train/LLaMA-Factory/output/deepseek_r1_1.5B_lora_sft/1.5B_correct_math500_solutions/maxcutoff_4096
# "/n/holylabs/LABS/sham_lab/Users/jbejjani/multiagent-reasoning/train/LLaMA-Factory/saves/deepseek-r1-1.5b/math500/full/sft" 
# --model_name_or_path "/n/holylabs/LABS/sham_lab/Users/jbejjani/DeepSeek-V3/models/DeepSeek-R1-Distill-Qwen-1.5B" \
# --model_name_or_path "/n/holylabs/LABS/sham_lab/Users/jbejjani/multiagent-reasoning/train/LLaMA-Factory/saves/deepseek-r1-1.5b/math500/full/sft" \