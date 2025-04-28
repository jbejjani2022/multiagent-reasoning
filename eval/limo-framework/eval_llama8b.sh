CUDA_VISIBLE_DEVICES='0' \
python eval.py \
--model_name_or_path "/n/netscratch/sham_lab/Everyone/jbejjani/output/llama3_lora_kto/llama8B_math500_solutions_kto" \
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

# /n/netscratch/sham_lab/Everyone/jbejjani/output/llama3_lora_kto/llama8B_math500_solutions_kto
# /n/netscratch/sham_lab/Everyone/jbejjani/output/llama3_lora_sft/math500_llama8B_correct_solutions
# --model_name_or_path "/n/holylabs/LABS/sham_lab/Users/jbejjani/Llama3/models/Meta-Llama-3-8B-Instruct" \