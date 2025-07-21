import requests
import json

class OllamaModel:
    def __init__(self, model_name="qwen3:8b", host="http://localhost:11434", additional_prompt=None):
        self.model_name = model_name
        self.model = model_name  # For compatibility with code expecting .model
        self.host = host
        self.additional_prompt = additional_prompt

    def generate(self, prompts, num_return_sequences=1, temperature=0.8, max_tokens=512, do_sample=True, hide_input=False, eos_token_id=None):
        results = []
        for prompt in prompts:
            if self.additional_prompt:
                prompt = f"{prompt}\n{self.additional_prompt}"
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            response = requests.post(f"{self.host}/api/generate", json=payload)
            response.raise_for_status()
            # print("Ollama raw response:", response.text)  # Debug print

            # Concatenate all 'response' fields
            answer = ""
            for line in response.text.strip().splitlines():
                try:
                    obj = json.loads(line)
                    answer += obj.get("response", "")
                    if obj.get("done", False):
                        break
                except Exception as e:
                    print(f"Error parsing Ollama line: {e}\nLine: {line}")
            results.append(answer.strip())
        # Mimic the .text attribute for compatibility
        class Result:
            def __init__(self, texts):
                self.text = texts
        return Result(results)