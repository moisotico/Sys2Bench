#!/bin/bash

commands=(
    "python methods/CoT/AQuA/inference.py --base_lm openai --temperature 0.8"
    "python methods/ToT/AQuA/inference.py --base_lm openai --depth_limit 10 --beam_size 10"
    "python methods/CoT/blocksworld/cot_inference.py --base_lm openai --temperature 0.8"
    "python methods/ToT/blocksworld/tot_inference.py --base_lm openai-4omini --depth_limit 2 --beam_size 10 --temperature 0.8 --search_algo beam --reward_aggregator mean"
    "python methods/CoT/gsm8k/inference.py --base_lm openai"
    "python methods/ToT/gsm8k/inference.py --base-lm openai-4omini --depth_limit 10 --beam_size 10"
    "python methods/CoT/prontoqa/cot_inference.py --base_lm openai --temperature 0.8"
    "python methods/ToT/prontoqa/tot_inference.py --base_lm openai-4omini --depth_limit 10 --beam_size 5"
    "python methods/CoT/tripplan/inference.py --base-lm openai --temperature 0.8"
    "python methods/ToT/tripplan/inference.py --base-lm openai-4omini --depth_limit 10 --beam_size 10"
    "python -m torch.distributed.run --nproc_per_node 1 --master_port=25678 methods/CoT/calendarplan/inference.py --model_dir openai --temperature 1.0 --base_lm openai --num_days 1"
    "python -m torch.distributed.run --nproc_per_node 1 --master_port=25678 methods/CoT/calendarplan/inference.py --model_dir openai --temperature 1.0 --base_lm openai --num_days 2"
)

for cmd in "${commands[@]}"; do
  echo "======================================"
  echo "Running: $cmd"
  
  timeout 30 bash -c "$cmd" >> ./outputs.log 2>&1
  exit_code=$?
  
  if [ $exit_code -eq 124 ]; then
    echo "Command timed out after 10 seconds."
  elif [ $exit_code -eq 1 ]; then
    echo "Command returned exit code 1."
  else
    echo "Command finished with exit code $exit_code."
  fi
  echo "======================================"
  echo ""
done
