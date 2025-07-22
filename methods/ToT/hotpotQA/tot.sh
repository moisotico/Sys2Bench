python -m torch.distributed.run --nproc_per_node 1 --master_port=25678 methods/ToT/hotpotQA/inference.py --model_dir openai --temperature 1.0 --base_lm openai --openai_model gpt-4o-mini
# Ollama
python -m torch.distributed.run --nproc_per_node 1 --master_port=25678 methods/ToT/hotpotQA/inference.py --base_lm ollama --model_name qwen3:8b --temperature 1.0 
