from typing import Optional, Literal
from datetime import datetime
import re
import torch
import json
import fire
from reasoners import LanguageModel, Reasoner
from reasoners.benchmark import BWEvaluator
from reasoners.algorithm import BeamSearch, DFS
from reasoners.lm import HFModel, LLaMaApiModel, OpenAIModel
from reasoners.lm.ollama_model import OllamaModel
from world_model import BlocksWorldModel
from search_config import BWConfig

def extract_step(file_path):
    match = re.search(r'_step_(\d+)_', file_path)
    return int(match.group(1)) if match else None

def bfs_bw_extractor(algo_output):
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    try:
        return "\n".join(algo_output.terminal_node.state.action_history)
    except Exception as e:
        print("Error in output extraction,", e)
        return ""
    
def dfs_bw_extractor(algo_output):
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    try:
        return "\n".join(algo_output.terminal_state.action_history)
    except Exception as e:
        print("Error in output extraction,", e)
        return ""

def tot_bw(base_model: LanguageModel,
           prompt: dict,
           search_algo: str = "beam",
           data_path: str = 'data',
           resume: int = 0,
           depth_limit: int = 6,
           log_dir: Optional[str] = None,
           disable_log: bool = False,
           domain_file: str = "",
           config_file: str = "",
           lm_plan_file: str = 'lm_plan.tmp',
           temperature: float = 0.8,
           calc_reward: Literal['sampling', 'logits'] = 'sampling',
           **search_algo_params):

    if search_algo == "beam":
        search_algo_params |= {"max_depth": depth_limit}
    elif search_algo == "dfs":
        search_algo_params |= {"depth": depth_limit}
    else:
        print("Unknown search algorithm", search_algo)
        raise NotImplementedError
    world_model = BlocksWorldModel(base_model=base_model, prompt=prompt, max_steps=depth_limit)
    config = BWConfig(base_model=base_model, prompt=prompt, temperature=temperature, calc_reward=calc_reward)
    
    output_extractor = dfs_bw_extractor if search_algo == "dfs" else bfs_bw_extractor
    if search_algo == "dfs":
        search_algo_instance = DFS(**search_algo_params)
    elif search_algo == "beam":
        search_algo_instance = BeamSearch(**search_algo_params)
    else:
        raise NotImplementedError
    
    reasoner = Reasoner(world_model=world_model, search_config=config, search_algo=search_algo_instance)
    evaluator = BWEvaluator(config_file=config_file, domain_file=domain_file, data_path=data_path, init_prompt=prompt, disable_log=disable_log, output_extractor=output_extractor)
    accuracy = evaluator.evaluate(reasoner, shuffle_prompt=True, num_shot=4, resume=resume, log_dir=log_dir)
    print(accuracy)

def main(
    base_lm: Literal['hf', 'openai', 'api', 'ollama']  = 'openai',
    model_dir = '/path/to/model',
    prompt_path: str = 'prompts/blocksworld/pool_prompt_v1.json',
    data_path: str = 'data/blocksworld/split_v1/split_v1_step_2_data.json',
    disable_log: bool = False,
    config_file: str = "data/blocksworld/bw_config.yaml",
    domain_file: str = "data/blocksworld/generated_domain.pddl",
    lm_plan_file: str = 'lm_plan.tmp',
    depth_limit: int = 6,
    mem_map = None,
    temperature = 0.8,
    search_algo = "beam",
    batch_size = 8,
    quantized = None,
    calc_reward: Literal['sampling', 'logits'] = 'sampling',
    api_model_id='meta-llama/Meta-Llama-3.1-70B-Instruct',
    openai_model='gpt-4o-mini',
    model_name=None,
    **kwargs
):
    with open(prompt_path) as f:
        prompt = json.load(f)

    # Model selection
    if base_lm == 'openai':
        base_model = OpenAIModel(openai_model, additional_prompt="CONTINUE")
    elif base_lm == 'api':
        base_model = LLaMaApiModel(None, None, use_api=True, model_id=api_model_id, quantized=None, additional_prompt="CONTINUE")
        model_dir = base_model.model_id
    elif base_lm == 'hf':
        print("Quantized: ", quantized)
        base_model = HFModel(model_pth=model_dir, tokenizer_pth=model_dir, quantized=quantized)
    elif base_lm == 'ollama':
        base_model = OllamaModel(model_name=model_name, additional_prompt=None)
    else:
        raise ValueError(f"base_lm {base_lm} is not supported")
    
    # Logging
    if base_lm == 'hf' or base_lm == 'api':
        model_name = model_dir.split('/')[-1]
    elif model_name:
        model_name = model_name
    else:
        model_name = base_lm
    log_dir =  f'logs/blocksworld/tot/{datetime.now().strftime("%m%d%Y-%H%M%S")}_{model_name}'

    tot_bw(
        base_model,
        prompt,
        search_algo=search_algo,
        disable_log=disable_log,
        data_path=data_path,
        config_file=config_file,
        domain_file=domain_file,
        depth_limit=depth_limit,
        lm_plan_file=lm_plan_file,
        temperature=temperature, 
        calc_reward=calc_reward,
        log_dir=log_dir,
        **kwargs
    )

if __name__ == '__main__':
    fire.Fire(main)
