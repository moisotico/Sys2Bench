import os
import logging
from typing import Optional, Union, Dict, Any
import time
import requests
import json

from .. import LanguageModel, GenerateOutput

logger = logging.getLogger(__name__)

class AWSBedrockClient:
    """
    Simplified AWS Bedrock client for the reasoner model.
    """
    
    def __init__(self, aws_region: str, bearer_token: str, timeout: int = 60):
        self.aws_region = aws_region
        self.bearer_token = bearer_token
        self.timeout = timeout
        self.base_url = f"https://bedrock-runtime.{aws_region}.amazonaws.com"
        
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {bearer_token}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
        logger.info(f"AWS Bedrock client initialized for region: {aws_region}")

    def _format_request_body(self, model_id: str, prompt: str, max_tokens: int = 2000, temperature: float = 0.2) -> Dict[str, Any]:
        """Format request body based on model type."""
        
        # Claude models
        if 'claude' in model_id.lower():
            return {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}]
            }
        
        # Llama models
        elif 'llama' in model_id.lower():
            return {
                "prompt": prompt,
                "max_gen_len": max_tokens,
                "temperature": temperature
            }
        
        # Mistral models
        elif 'mistral' in model_id.lower():
            return {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
        
        # Titan models
        elif 'titan' in model_id.lower():
            return {
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": max_tokens,
                    "temperature": temperature
                }
            }
        
        # Default format
        return {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature
        }

    def _extract_response_text(self, response_body: Dict[str, Any], model_id: str) -> str:
        """Extract text from response based on model type."""
        
        try:
            # Claude models
            if 'claude' in model_id.lower():
                return response_body['content'][0]['text']
            
            # Llama models
            elif 'llama' in model_id.lower():
                return (response_body.get('generation') or 
                       response_body.get('outputs', [{}])[0].get('text', '') or
                       response_body.get('choices', [{}])[0].get('text', ''))
            
            # Mistral models
            elif 'mistral' in model_id.lower():
                return (response_body.get('outputs', [{}])[0].get('text', '') or
                       response_body.get('choices', [{}])[0].get('message', {}).get('content', ''))
            
            # Titan models
            elif 'titan' in model_id.lower():
                return response_body.get('results', [{}])[0].get('outputText', '')
            
            # Generic fallback
            return (response_body.get('completion') or 
                   response_body.get('generated_text') or 
                   response_body.get('text') or 
                   response_body.get('content') or
                   str(response_body))
                       
        except (KeyError, IndexError, TypeError) as e:
            logger.warning(f"Could not extract text from response for {model_id}: {e}")
            return str(response_body)

    def generate(self, model_id, prompt, options=None, timeout=60) -> Dict[str, Any]:
        """Generate a response using AWS Bedrock."""
        
        max_tokens = options.get('max_tokens', 2000) if options else 2000
        temperature = options.get('temperature', 0.2) if options else 0.2

        request_body = self._format_request_body(model_id, prompt, max_tokens, temperature)
        
        try:
            url = f"{self.base_url}/model/{model_id}/invoke"
            
            response = self.session.post(
                url,
                json=request_body,
                timeout=timeout if timeout else self.timeout
            )
            
            if response.status_code == 401:
                raise PermissionError("Authentication failed. Check your bearer token.")
            elif response.status_code == 403:
                raise PermissionError(f"Access denied for model {model_id}. Check your permissions.")
            elif response.status_code == 404:
                raise ValueError(f"Model {model_id} not found or endpoint not available.")
            elif response.status_code == 429:
                raise Exception("Rate limit exceeded. Please try again later.")
            elif not response.ok:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                raise Exception(f"Request failed: {error_msg}")
            
            try:
                response_data = response.json()
            except json.JSONDecodeError:
                raise Exception(f"Invalid JSON response: {response.text}")
            
            if 'error' in response_data:
                error_msg = response_data.get('error', {}).get('message', 'Unknown error')
                raise Exception(f"Bedrock API error: {error_msg}")
            
            result_text = self._extract_response_text(response_data, model_id)
            
            return {
                'response': result_text,
                'provider': 'aws_bedrock',
                'model_id': model_id,
                'usage': response_data.get('usage', {}),
                'raw_response': response_data
            }
            
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Request timed out after {timeout} seconds")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Failed to connect to {self.base_url}")
        except Exception as e:
            logger.error(f"Error in AWS Bedrock client: {e}")
            raise e

additional_prompt_templates = {
    'ANSWER': "Your response need to be ended with \"So the answer is <Answer>\"\n\n",
    'CONTINUE': "Please continue to answer the last question, following the format of previous examples. Don't say any other words.\n\n",
    'NONE': ""
}

PROMPT_TEMPLATE_ANSWER = "Your response need to be ended with \"So the answer is\"\n\n"
PROMPT_TEMPLATE_CONTINUE = "Please continue to answer the last question, following the format of previous examples. Don't say any other words.\n\n"
PROMPT_TEMPLATE_CUSTOM = "You are required to generate all possible actions at a given state. Please continue to answer the last question, following the format of previous examples. Don't say any other words.\n\n"

class AWSBedrockModel(LanguageModel):
    def __init__(self, 
                 model_id: str = "meta.llama3-1-8b-instruct-v1:0",
                 aws_region: str = "us-east-1",
                 bearer_token: str = None,
                 max_tokens: int = 2048, 
                 temperature: float = 0.0, 
                 additional_prompt: str = None,
                 timeout: int = 60):
        """
        Initialize AWS Bedrock model.
        
        Args:
            model_id: The AWS Bedrock model ID to use
            aws_region: AWS region for Bedrock service
            bearer_token: Bearer token for authentication
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            additional_prompt: Additional prompt template to use
            timeout: Request timeout in seconds
        """
        self.model_id = model_id
        self.model = model_id  # For compatibility with code expecting .model
        self.aws_region = aws_region
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.additional_prompt = additional_prompt
        self.timeout = timeout
        
        # Get bearer token from environment if not provided
        if bearer_token is None:
            bearer_token = os.getenv("AWS_BEDROCK_BEARER_TOKEN")
        
        if not bearer_token:
            raise ValueError("Bearer token must be provided either as parameter or AWS_BEDROCK_BEARER_TOKEN environment variable")
        
        # Initialize the AWS Bedrock client
        self.client = AWSBedrockClient(
            aws_region=aws_region,
            bearer_token=bearer_token,
            timeout=timeout
        )
        
        self.tokens_generated = 0
        logger.info(f"AWS Bedrock model initialized: {model_id} in region {aws_region}")

    def generate(self,
                prompt: Optional[Union[str, list[str]]],
                max_tokens: int = None,
                top_p: float = 1.0,
                num_return_sequences: int = 1,
                rate_limit_per_min: Optional[int] = 20,
                stop: Optional[str] = None,
                temperature = None,
                additional_prompt = None,
                retry = 64,
                custom_prompt_template: str = "",
                system_prompt = None,
                **kwargs) -> GenerateOutput:
        """
        Generate text using AWS Bedrock model.
        
        Args:
            prompt: Input prompt(s)
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            num_return_sequences: Number of sequences to return
            rate_limit_per_min: Rate limiting (requests per minute)
            stop: Stop sequences
            temperature: Sampling temperature
            additional_prompt: Additional prompt template
            retry: Number of retry attempts
            custom_prompt_template: Custom prompt template
            system_prompt: System prompt (if supported by model)
            
        Returns:
            GenerateOutput containing generated text
        """
        
        temperature = self.temperature if temperature is None else temperature
        
        # Handle prompt input format
        if isinstance(prompt, list):
            if len(prompt) == 1:
                prompt = prompt[0]
            else:
                # For multiple prompts, process them sequentially
                results = []
                for p in prompt:
                    single_result = self.generate(
                        p, max_tokens=max_tokens, top_p=top_p, 
                        num_return_sequences=1, rate_limit_per_min=rate_limit_per_min,
                        stop=stop, temperature=temperature, 
                        additional_prompt=additional_prompt, retry=retry,
                        custom_prompt_template=custom_prompt_template,
                        system_prompt=system_prompt, **kwargs
                    )
                    results.extend(single_result.text)
                return GenerateOutput(text=results, log_prob=None)
        
        # Handle additional prompt templates
        if additional_prompt is None and self.additional_prompt is not None:
            additional_prompt = self.additional_prompt
            
        # Apply prompt templates
        if additional_prompt == "ANSWER":
            prompt = PROMPT_TEMPLATE_ANSWER + prompt
        elif additional_prompt == "CONTINUE":
            prompt = PROMPT_TEMPLATE_CONTINUE + prompt
        elif additional_prompt == "CUSTOM":
            prompt = PROMPT_TEMPLATE_CUSTOM + prompt
        elif additional_prompt == "NONE":
            prompt = prompt
        elif custom_prompt_template != "":
            prompt = custom_prompt_template + prompt

        if max_tokens is None:
            max_tokens = self.max_tokens

        # Prepare options for the client
        options = {
            'max_tokens': max_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'stop': stop
        }
        
        # Add system prompt if provided and supported
        if system_prompt is not None:
            options['system_prompt'] = system_prompt

        # Retry logic
        for i in range(1, retry + 1):
            try:
                # Rate limiting
                if rate_limit_per_min is not None:
                    time.sleep(60 / rate_limit_per_min)
                
                # Generate response using AWS Bedrock client
                response = self.client.generate(
                    model_id=self.model_id,
                    prompt=prompt,
                    options=options,
                    timeout=self.timeout
                )
                
                # Extract generated text
                generated_text = response.get('response', '')
                
                # Handle multiple return sequences (simulate by repeating the call)
                results = []
                for _ in range(num_return_sequences):
                    if _ == 0:
                        # Use the original response for the first sequence
                        results.append(generated_text)
                    else:
                        # For additional sequences, make new calls with slight temperature variation
                        temp_options = options.copy()
                        temp_options['temperature'] = min(1.0, temperature + 0.1 * _)
                        temp_response = self.client.generate(
                            model_id=self.model_id,
                            prompt=prompt,
                            options=temp_options,
                            timeout=self.timeout
                        )
                        results.append(temp_response.get('response', ''))
                
                # Update token count if available
                if 'usage' in response:
                    usage = response['usage']
                    if 'completion_tokens' in usage:
                        self.tokens_generated += usage['completion_tokens']
                    elif 'output_tokens' in usage:
                        self.tokens_generated += usage['output_tokens']
                
                return GenerateOutput(
                    text=results,
                    log_prob=None  # AWS Bedrock doesn't typically return log probabilities
                )
                
            except Exception as e:
                logger.warning(f"AWS Bedrock generation attempt {i} failed: {e}")
                if i < retry:
                    time.sleep(i)  # Exponential backoff
                else:
                    raise e
        
        # After all retries failed
        raise RuntimeError(f"AWS Bedrock model failed to generate output after {retry} attempts")

    def get_next_token_logits(self,
                              prompt: Union[str, list[str]],
                              candidates: Union[list[str], list[list[str]]],
                              **kwargs):
        """AWS Bedrock models do not support next token logits."""
        raise NotImplementedError("AWS Bedrock models do not support get_next_token_logits")

    def get_loglikelihood(self,
                          prompt: Union[str, list[str]],
                          **kwargs):
        """AWS Bedrock models do not support log likelihood calculation."""
        raise NotImplementedError("AWS Bedrock models do not support get_loglikelihood")


# For compatibility with existing code that might expect OllamaModel-like interface
class BedrockModel:
    """Alternative interface similar to OllamaModel for backward compatibility."""
    
    def __init__(self, 
                 model_name: str = "meta.llama3-1-8b-instruct-v1:0",
                 aws_region: str = "us-east-1", 
                 bearer_token: str = None,
                 additional_prompt: str = None):
        self.model_name = model_name
        self.model = model_name
        self.aws_region = aws_region
        self.additional_prompt = additional_prompt
        
        # Get bearer token from environment if not provided
        if bearer_token is None:
            bearer_token = os.getenv("AWS_BEDROCK_BEARER_TOKEN")
        
        if not bearer_token:
            raise ValueError("Bearer token must be provided either as parameter or AWS_BEDROCK_BEARER_TOKEN environment variable")
        
        self.client = AWSBedrockClient(
            aws_region=aws_region,
            bearer_token=bearer_token
        )

    def generate(self, prompts, num_return_sequences=1, temperature=0.8, max_tokens=512, 
                 do_sample=True, hide_input=False, eos_token_id=None):
        """Generate method compatible with OllamaModel interface."""
        results = []
        
        if isinstance(prompts, str):
            prompts = [prompts]
            
        for prompt in prompts:
            if self.additional_prompt:
                prompt = f"{prompt}\n{self.additional_prompt}"
                
            options = {
                'max_tokens': max_tokens,
                'temperature': temperature
            }
            
            try:
                response = self.client.generate(
                    model_id=self.model_name,
                    prompt=prompt,
                    options=options
                )
                results.append(response.get('response', '').strip())
                
            except Exception as e:
                logger.error(f"Error generating with AWS Bedrock: {e}")
                results.append("")
        
        # Mimic the .text attribute for compatibility
        class Result:
            def __init__(self, texts):
                self.text = texts
                
        return Result(results)


if __name__ == '__main__':
    # Example usage
    try:
        model = AWSBedrockModel(
            model_id="meta.llama3-1-8b-instruct-v1:0",
            aws_region="us-east-1",
            bearer_token=os.getenv("AWS_BEDROCK_BEARER_TOKEN")
        )
        
        result = model.generate(['Hello, how are you?'])
        print("Generated text:", result.text)
        
    except Exception as e:
        print(f"Error testing AWS Bedrock model: {e}")
