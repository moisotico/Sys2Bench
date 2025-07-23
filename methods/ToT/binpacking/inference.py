import pickle
from typing import Type, Optional, Literal
import json
import numpy as np
from tqdm import tqdm
from datetime import datetime

from reasoners import LanguageModel, Reasoner, SearchAlgorithm
from reasoners.algorithm import BeamSearch, MCTS, MCTSNode
from reasoners.benchmark import BinPackingEvaluator

from world_model import BinPackingWorldModel, BinPackingState, BinPackingAction
from search_config import BinPackingConfig
import utils


def tot_game24(base_model: LanguageModel,
               prompts: dict,
               search_algo: Type[SearchAlgorithm] = BeamSearch,
               resume: int = 0,
               n_action: int = 4,
               n_beam: int = 5,
               n_eval: int = 3,
               depth_limit: int = 8,
               batch_size: int = 3,
               log_dir: Optional[str] = None,
               disable_log: bool = False,
               calc_reward: Literal['sampling', 'logits'] = 'sampling',
               temperature: float = 0.7,
               **search_algo_params):
    ## keep the best 5 candidates, need at most 4 steps to solve
    ## following ToT, eval step will consider number of times to prompt for state evaluation
    # search_algo_params |= {'beam_size': n_beam, 'max_depth': depth_limit}
    
    if search_algo.__name__ == 'BeamSearch':
        search_algo_params['beam_size'] = n_beam
        search_algo_params['max_depth'] = depth_limit
    search_algo_params |= {'output_trace_in_each_iter': True, 'depth_limit': depth_limit, 'disable_tqdm': False}
    world_model = BinPackingWorldModel(base_model=base_model, prompt=prompts, batch_size=batch_size)
    config = BinPackingConfig(base_model=base_model, prompt=prompts, calc_reward=calc_reward,
                          n_actions=n_action, n_eval=n_eval, batch_size=batch_size, depth_limit=depth_limit, temperature=temperature)
    search_algo = search_algo(**search_algo_params)
    reasoner = Reasoner(world_model=world_model, search_config=config, search_algo=search_algo)

    evaluator = BinPackingEvaluator(
        output_extractor=utils.parse_output,
        answer_extractor=lambda x: x,
        init_prompt=prompts, # will update dynamically
        disable_log=False,
        disable_tqdm=False,
        sample_prompt_type="tot", # adapt to RAP as well.
    )
    
    metric = evaluator.evaluate(reasoner, shuffle_prompt=True, num_shot=5, resume=resume, log_dir=log_dir)
    print(f'Acc: {metric}')


if __name__ == '__main__':
    import fire

    def main(base_lm: Literal['hf', 'openai', 'api', 'ollama'] = 'openai',
             model_dir: str = 'meta-llama/Llama-2-13b-hf',
             quantized: Optional[Literal['awq', 'int8', 'fp4', 'nf4']] = None,
             batch_size: int = 1,
             prompts: str = 'prompts/binpacking/prompts.json',
             disable_log: bool = False,
             disable_tqdm: bool = False,
             search_algo: Literal['beamsearch', 'mcts'] = 'beamsearch',
             temperature: float=0.8,
             depth_limit: int = 8,
             api_model_id='meta-llama/Meta-Llama-3.1-70B-Instruct',
             openai_model='gpt-4o-mini',
             calc_reward: Literal['sampling', 'logits'] = 'sampling',
             model_name=None,
             **kwargs):
        
        with open(prompts) as f:
            prompts = json.load(f)

        if base_lm == 'hf':
            from reasoners.lm import HFModel
            base_model = HFModel(model_dir, model_dir, max_batch_size=batch_size, max_new_tokens=512,
                                 quantized=quantized, temperature=temperature)
        elif base_lm == 'openai':
            from reasoners.lm.openai_model import OpenAIModel
            base_model = OpenAIModel(openai_model, additional_prompt="NONE", temperature=temperature)
        elif base_lm == 'api':
            from reasoners.lm.llama_api_model import LLaMaApiModel
            base_model = LLaMaApiModel(None, None, use_api=True, model_id=api_model_id, quantized=None)
            model_dir = base_model.model_id
        elif base_lm == 'ollama':
            from reasoners.lm.ollama_model import OllamaModel
            if model_name is None:
                model_name = "qwen3:8b"
            base_model = OllamaModel(model_name=model_name, additional_prompt="NONE")
        else:
            raise ValueError(f"base_lm {base_lm} is not supported")
            
        if base_lm == 'hf' or base_lm == 'api':
            model_name = model_dir.split('/')[-1]
        else:
            model_name = f'{base_lm}_{base_model.model}'
        
        if search_algo == 'beamsearch':
            search_algo = BeamSearch
        elif search_algo == 'mcts':
            search_algo = MCTS
        
        log_dir =  f'logs/binpacking/'\
                        f'{search_algo.__name__}/'\
                        f'{datetime.now().strftime("%m%d%Y-%H%M%S")}'    
        
        log_dir = log_dir + '_' + model_name
        
        tot_game24(base_model=base_model,
                   prompts=prompts,
                   batch_size=batch_size,
                   n_beam=5,
                   disable_log=disable_log,
                   log_dir=log_dir,
                   search_algo=search_algo,
                   depth_limit=depth_limit,
                   temperature=temperature,
                   calc_reward=calc_reward,
                   **kwargs)


    fire.Fire(main)
