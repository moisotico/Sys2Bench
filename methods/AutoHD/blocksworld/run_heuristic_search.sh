export CUDA_VISIBLE_DEVICES=0
# export base_lm="openai" 
export base_lm="together-ai"

export log_name="demo"
export VAL="/mnt/data/shared/shparashar/Reasoning/planner_tools/VAL"
export PR2="/mnt/data/shared/shparashar/Reasoning/planner_tools/PR2"
export quantized=None

python examples/A-star/blocksworld/heuristic_search.py \
--data_path 'examples/CoT/blocksworld/data/subset/val_set_V3.json'   \
--test_file_path 'examples/CoT/blocksworld/data/split_v1/split_v1_step_2_data.json' \
--base_lm $base_lm \
--depth_limit 20 \
--prompt_path examples/CoT/blocksworld/prompts/pool_prompt_v1.json \
--beam_size 10 
