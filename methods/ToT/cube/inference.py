import json
from typing import Literal
from datetime import datetime

import fire
import utils
from search_config import CubeConfig
from world_model import CubeWorldModel

from reasoners import Reasoner
from reasoners.algorithm import BeamSearch
from reasoners.benchmark import CubeEvaluator
from reasoners.lm.hf_model import HFModel
from reasoners.lm.llama_api_model import LLaMaApiModel
from reasoners.lm.openai_model import OpenAIModel
from reasoners.lm.ollama_model import OllamaModel


def main(
    base_lm: Literal["hf", "openai", "api", "ollama"] = "openai",
    model_dir=None,
    api_model_id="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    openai_model="gpt-4o-mini",
    prompt="prompts/cube/prompts.json",
    resume=0,
    log_dir=None,
    temperature=0.8,
    calc_reward="llm",
    quantized="int8",
    depth_limit: int = 10,
    beam_size: int = 5,
    model_name=None,
    **search_algo_params,
):
    if base_lm == "openai":
        base_model = OpenAIModel(openai_model, additional_prompt="ANSWER")
    elif base_lm == "hf":
        base_model = HFModel(
            model_dir, model_dir, quantized=quantized, additional_prompt="ANSWER"
        )
    elif base_lm == "api":
        base_model = LLaMaApiModel(
            None,
            None,
            use_api=True,
            model_id=api_model_id,
            quantized=None,
            additional_prompt="ANSWER",
        )
        model_dir = base_model
    elif base_lm == "ollama":
        base_model = OllamaModel(model_name=model_name or "qwen3:8b", additional_prompt=None)
    else:
        raise ValueError(f"Unknown base_lm: {base_lm}")

    with open(prompt) as f:
        prompt = json.load(f)

    # Consistent model_name for logging
    if model_name:
        log_model_name = model_name
    elif base_lm == "hf" or base_lm == "api":
        log_model_name = model_dir.split("/")[-1] if model_dir else base_lm
    else:
        log_model_name = base_lm

    log_dir = f"logs/cube/ToT/{datetime.now().strftime('%m%d%Y-%H%M%S')}_{log_model_name}"

    search_algo_params |= {"max_depth": depth_limit}
    search_algo_params |= {
        "sampling_strategy": "argmax",
        "reward_aggregator": "mean",
        "beam_size": beam_size,
    }

    world_model = CubeWorldModel(base_model=base_model, prompt=prompt)
    config = CubeConfig(
        base_model=base_model,
        prompt=prompt,
        temperature=temperature,
        calc_reward=calc_reward,
    )
    search_algo = BeamSearch(**search_algo_params)

    reasoner = Reasoner(
        world_model=world_model, search_config=config, search_algo=search_algo
    )
    evaluator = CubeEvaluator(
        output_extractor=utils.tot_extractor,
        answer_extractor=utils.retrieve_answer_from_dataset,
        init_prompt=prompt,
        disable_log=False,
        disable_tqdm=False,
        sample_prompt_type="tot",
    )

    accuracy = evaluator.evaluate(
        reasoner, shuffle_prompt=True, num_shot=4, resume=resume, log_dir=log_dir
    )
    print(f"accuracy: {accuracy:.4f}")
    return 0


if __name__ == "__main__":
    fire.Fire(main)
