export CUDA_VISIBLE_DEVICES=0
export base_lm="openai" 
# export base_lm="together-ai"
export log_name="demo"
export VAL="/mnt/data/shared/shparashar/Reasoning/planner_tools/VAL"
export PR2="/mnt/data/shared/shparashar/Reasoning/planner_tools/PR2"
export quantized=None

python examples/A-star/blocksworld/Astar_inference.py \
--data_path 'examples/CoT/blocksworld/data/split_v1/split_v1_step_4_data.json' \
--base_lm $base_lm \
--depth_limit 4 \
--prompt_path examples/CoT/blocksworld/prompts/pool_prompt_v1.json \
--n_iters 20 \
--lm_plan_file 'lm_plan.tmp'
