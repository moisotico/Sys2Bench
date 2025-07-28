from typing import Literal
import json
from reasoners.benchmark import AQuAEvaluator
import utils
import fire
from tqdm import tqdm
from reasoners.lm.hf_model import HFModel
from reasoners.lm.openai_model import OpenAIModel
from reasoners.lm.llama_api_model import LLaMaApiModel
from reasoners.lm.ollama_model import OllamaModel
from reasoners.lm.aws_bedrock_model import AWSBedrockModel

class CoTReasoner():
    def __init__(self, base_model, temperature=0.8, sc_num = 1):
        self.base_model = base_model
        self.temperature = temperature
        self.sc_num = sc_num
        
    def __call__(self, example, prompt=None):
        inputs = prompt["cot"].replace("{QUESTION}", example)
        outputs = []
        do_sample = True
        if self.temperature == 0 and isinstance(self.base_model, HFModel):
            print("Using greedy decoding with HF model. Set do_sample=False")
            self.temperature == 1.0
            do_sample = False
        if isinstance(self.base_model, OpenAIModel) or isinstance(self.base_model, LLaMaApiModel):
            eos_token_id = []
        elif isinstance(self.base_model, OllamaModel) or isinstance(self.base_model, AWSBedrockModel):
            eos_token_id = [1]
        else:
            print(self.base_model.model.__class__)
            print(self.base_model.model.config.architectures[0])
            eos_token_id = [13]
        
        for _ in tqdm(range(self.sc_num), leave=False):
            gen_kwargs = {
                "hide_input": True,
                "do_sample": do_sample,
                "temperature": self.temperature,
                "eos_token_id": eos_token_id
            }
            # Only add additional_prompt for non-Ollama and non-AWSBedrock models
            if not isinstance(self.base_model, (OllamaModel, AWSBedrockModel)):
                gen_kwargs["additional_prompt"] = "ANSWER"
            output = self.base_model.generate([inputs], **gen_kwargs).text[0].strip()
            outputs.append(output)
        
        return outputs
            
            

def main(base_lm:Literal['hf', 'openai', 'api', 'ollama', 'aws_bedrock'] = 'openai',
         model_dir= None, 
         prompt="prompts/aqua/prompts.json", 
         data_path="data/aqua", 
         datasetname="test",
         quantized='int8',
         resume=0, 
         temperature=0,
         sc_num=1,
         api_model_id='meta-llama/Meta-Llama-3.1-8B-Instruct',
         model_name=None,
         openai_model="gpt-4o-mini",
         log_dir=None,
         aws_region="us-east-1",
         bearer_token=None):

    if base_lm == "openai":
        base_model = OpenAIModel(openai_model, additional_prompt="ANSWER")
    elif base_lm == "hf":
        base_model = HFModel(model_dir, model_dir,quantized=quantized)
    elif base_lm == 'api':
        base_model = LLaMaApiModel(None, None, use_api=True, model_id=api_model_id, quantized=None, additional_prompt="ANSWER")
        model_dir = base_model.model_id
    elif base_lm == 'ollama':
        if model_name is None:
            model_name = "qwen3:8b"
        base_model = OllamaModel(model_name=model_name, additional_prompt="ANSWER")
    elif base_lm == 'aws_bedrock':
        base_model = AWSBedrockModel(
            model_id=model_name or "meta.llama3-1-8b-instruct-v1:0",
            aws_region=aws_region,
            bearer_token=bearer_token,
            additional_prompt="ANSWER"
        )
    else:
        raise ValueError(f"base_lm {base_lm} is not supported")
    with open(prompt) as f:
        prompt = json.load(f)

    reasoner = CoTReasoner(base_model, temperature=temperature, sc_num=sc_num)
    
    output_extractor = utils.retrieve_answer
    if sc_num > 1:
        output_extractor = utils.cot_sc_extractor
    
    evaluator = AQuAEvaluator(
                 output_extractor=output_extractor,
                 answer_extractor=lambda x: utils.retrieve_answer_from_dataset(x["answer"]),
                 init_prompt=prompt, # will update dynamically
                 disable_log=False,
                 disable_tqdm=False,
                 sample_prompt_type="cot",
                 dataset_path=data_path,
                 datasetname=datasetname)
    from datetime import datetime
    log_dir =  f'logs/AQuA/'\
                        f'cot_{sc_num}/'\
                        f'{datetime.now().strftime("%m%d%Y-%H%M%S")}'
    
    if base_lm == 'hf' or base_lm == 'api':
        model_name= model_dir.split('/')[-1]
    else:
        model_name = base_lm
    log_dir = log_dir + f'_{model_name}'

    accuracy = evaluator.evaluate(reasoner, shuffle_prompt=True, num_shot=4, resume=resume, log_dir=log_dir)
    print(f'accuracy: {accuracy:.4f}')
    return 0



if __name__ == '__main__':
    fire.Fire(main)
