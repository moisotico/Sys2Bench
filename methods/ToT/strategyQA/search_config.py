from reasoners import LanguageModel, SearchConfig
from world_model import SQAState, SQAAction
from typing import Literal

class SQAConfig(SearchConfig):
    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict,
                 temperature: float = 0.8,
                 n_candidate: int = 3,
                 depth_limit: int = 10,
                 calc_reward: Literal['sampling', 'logits'] = 'logits') -> None:
        super().__init__()
        self.base_model = base_model
        self.example = None
        self.prompt = prompt
        self.n_candidate = n_candidate
        self.temperature = temperature
        self.depth_limit = depth_limit
        self.calc_reward = calc_reward

    def get_actions(self, state: SQAState) -> list[SQAAction]:
        input_prompt = self.prompt['tot_prefix']
        input_prompt += self.prompt['tot']
        input_prompt = input_prompt.replace("{QUESTION}", self.example)
        input_prompt += "".join([" " + s for s in state.state_history])

        outputs = []
        if len(state.state_history) == self.depth_limit - 1:
            input_prompt += self.prompt['propose_answer_instructions']
            outputs = self.base_model.generate(
                [input_prompt],
                num_return_sequences=self.n_candidate,
                temperature=self.temperature,
                do_sample=True,
                hide_input=True
            ).text
        else:
            input_prompt += self.prompt['propose_action_instructions']
            generate_kwargs = {
                "num_return_sequences": self.n_candidate,
                "temperature": self.temperature,
                "do_sample": True,
                "hide_input": True,
            }
            # Only add these if not OllamaModel
            if self.base_model.__class__.__name__ != "OllamaModel":
                generate_kwargs["stop"] = "."
                generate_kwargs["additional_prompt"] = "CONTINUE"
            outputs = self.base_model.generate([input_prompt], **generate_kwargs).text

        print("Action history: ", "".join(["  " + s for s in state.state_history]))
        print("-------------------------------")
        ret = list(dict.fromkeys(outputs))
        print("ACTIONS Outputs: with temperature  ", self.temperature,"  :  ", ret)
        return ret


    def fast_reward(self, state: SQAState, action: SQAAction) -> tuple[float, dict]:
        input_prompt = self.prompt['tot_prefix']
        input_prompt += self.prompt['tot']
        input_prompt = input_prompt.replace("{QUESTION}", self.example)
        input_prompt += "".join(["\n" + s for s in state.state_history])

        # Handle OllamaModel separately
        if self.base_model.__class__.__name__ == "OllamaModel":
            rating_prompt = self.prompt['rating_prompt'].replace("{input_prompt}", input_prompt).replace("{action}", action)
            generate_kwargs = {
                "temperature": self.temperature,
                "do_sample": True,
                "hide_input": True,
            }
            try:
                rating_response = self.base_model.generate([rating_prompt], **generate_kwargs)
                rating = float(rating_response.text[0].strip())
            except Exception:
                rating = 0
            return rating, {'intuition': rating, 'self_eval': rating}

        # Non-Ollama models
        if self.calc_reward == "sampling":   
            rating_prompt = self.prompt['rating_prompt'].replace("{input_prompt}", input_prompt).replace("{action}", action)
            rating_response = self.base_model.generate([rating_prompt], temperature=self.temperature)
            try:
                rating = float(rating_response.text[0].strip())
            except:
                rating = 0
            return rating, {'intuition': rating, 'self_eval': rating}
        
        intuition = self.base_model.get_loglikelihood(input_prompt + "\n", [input_prompt + "\n" + action])[0]
        self_eval_prompt = self.prompt['cot_prefix'] + self.prompt["self-eval"].replace("{Question}", self.example)
        self_eval_prompt += "\nACTION:\n" + action + "\nEVALUATION:\n"
        self_eval = self.base_model.get_loglikelihood(self_eval_prompt, 
            [self_eval_prompt + "good"])[0]
        
        print('Fast Reward Logits', self_eval, intuition)
        return intuition + self_eval, {'intuition': intuition, "self_eval": self_eval}

    def reward(self, state: SQAState, action: SQAAction, **kwargs) -> tuple[float, dict]:
        intuition, self_eval = kwargs['intuition'], kwargs['self_eval']
        return self_eval, {'intuition': intuition, "self_eval": self_eval}

    def update_example(self, example, prompt=None) -> None:
        super().update_example(example, prompt=prompt)