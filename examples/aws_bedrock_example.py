#!/usr/bin/env python3
"""
Example usage of the AWS Bedrock model in Sys2Bench.

This script demonstrates how to use the AWSBedrockModel for reasoning tasks.
"""

import os
import sys
from reasoners.lm import AWSBedrockModel, BedrockModel

# Add the reasoners path to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def main():
    """Example usage of AWS Bedrock models."""
    
    # Set your bearer token as an environment variable
    # export AWS_BEDROCK_BEARER_TOKEN="your_bearer_token_here"
    
    print("AWS Bedrock Model Example")
    print("-" * 30)
    
    try:
        # Example 1: Using AWSBedrockModel (full LanguageModel interface)
        print("1. Creating AWSBedrockModel...")
        model = AWSBedrockModel(
            model_id="meta.llama3-1-8b-instruct-v1:0",  # Llama 3.1 8B
            aws_region="us-east-1",
            bearer_token=os.getenv("AWS_BEDROCK_BEARER_TOKEN"),
            max_tokens=1024,
            temperature=0.7
        )
        
        print("2. Generating response...")
        result = model.generate(
            prompt="What is the capital of France? Explain briefly.",
            max_tokens=100,
            temperature=0.3
        )
        print(f"Response: {result.text[0]}")
        print(f"Tokens generated so far: {model.tokens_generated}")
        
        # Example 2: Using with additional prompt templates
        print("\n3. Using with ANSWER template...")
        result_with_template = model.generate(
            prompt="What is 2 + 2?",
            additional_prompt="ANSWER",
            max_tokens=50
        )
        print(f"Response with template: {result_with_template.text[0]}")
        
        # Example 3: Using BedrockModel (OllamaModel-like interface)
        print("\n4. Creating BedrockModel (simplified interface)...")
        simple_model = BedrockModel(
            model_name="meta.llama3-1-8b-instruct-v1:0",
            aws_region="us-east-1",
            bearer_token=os.getenv("AWS_BEDROCK_BEARER_TOKEN")
        )
        
        print("5. Generating with simple interface...")
        simple_result = simple_model.generate(
            prompts=["Hello! How are you today?"],
            temperature=0.5,
            max_tokens=100
        )
        print(f"Simple interface response: {simple_result.text[0]}")
        
        # Example 4: Multiple prompts
        print("\n6. Processing multiple prompts...")
        multi_result = model.generate(
            prompt=["What is AI?", "What is machine learning?"],
            max_tokens=50,
            num_return_sequences=1
        )
        print(f"AI response: {multi_result.text[0]}")
        print(f"ML response: {multi_result.text[1]}")
        
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Make sure to set AWS_BEDROCK_BEARER_TOKEN environment variable")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Check your AWS Bedrock configuration and bearer token")

def test_different_models():
    """Test different AWS Bedrock models."""
    
    models_to_test = [
        "meta.llama3-1-8b-instruct-v1:0",             # Llama 3.1 8B
        "us.meta.llama3-2-1b-instruct-v1:0",          # Llama 3.2 1B
        "us.meta.llama3-2-3b-instruct-v1:0",          # Llama 3.2 3B
        "us.meta.llama3-3-70b-instruct-v1:0",         # Llama 3.3 70B
        "us.meta.llama4-maverick-17b-instruct-v1:0",  # Llama 4 Maverick 17B
    ]
    
    bearer_token = os.getenv("AWS_BEDROCK_BEARER_TOKEN")
    if not bearer_token:
        print("Please set AWS_BEDROCK_BEARER_TOKEN environment variable")
        return
    
    test_prompt = "Explain quantum computing in one sentence."
    
    for model_id in models_to_test:
        try:
            print(f"\nTesting {model_id}...")
            model = AWSBedrockModel(
                model_id=model_id,
                aws_region="us-east-1",
                bearer_token=bearer_token,
                max_tokens=100,
                temperature=0.3
            )
            
            result = model.generate(prompt=test_prompt, max_tokens=100)
            print(f"Response: {result.text[0][:200]}...")
            
        except Exception as e:
            print(f"Error with {model_id}: {e}")

if __name__ == "__main__":
    print("Choose an option:")
    print("1. Basic usage example")
    print("2. Test different models")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        main()
    elif choice == "2":
        test_different_models()
    else:
        print("Invalid choice. Running basic example...")
        main()
