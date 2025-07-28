#!/usr/bin/env python3
"""
Script to automatically update all inference.py files to support AWS Bedrock models.
"""

import os
import re
from pathlib import Path

def update_inference_file(file_path):
    """Update a single inference.py file to support AWS Bedrock."""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 1. Add AWS Bedrock import if not present
        if 'from reasoners.lm.aws_bedrock_model import AWSBedrockModel' not in content:
            # Find the last lm import
            lm_imports = re.findall(r'from reasoners\.lm\.[^\s]+ import [^\n]+', content)
            if lm_imports:
                last_import = lm_imports[-1]
                content = content.replace(
                    last_import,
                    last_import + '\nfrom reasoners.lm.aws_bedrock_model import AWSBedrockModel'
                )
        
        # 2. Update Literal type hints to include aws_bedrock
        def add_aws_bedrock_to_literal(match):
            literal_content = match.group(1)
            if 'aws_bedrock' not in literal_content:
                return f'Literal[{literal_content.rstrip("\"")}, "aws_bedrock"]'
            return match.group(0)
        
        content = re.sub(r'Literal\[([^\]]*)\]', add_aws_bedrock_to_literal, content)
        
        # 3. Update eos_token_id conditions to include AWSBedrockModel
        # Pattern for OllamaModel only
        content = re.sub(
            r'elif isinstance\(self\.base_model, OllamaModel\):\s*\n\s*eos_token_id = \[1\]',
            'elif isinstance(self.base_model, OllamaModel) or isinstance(self.base_model, AWSBedrockModel):\n            eos_token_id = [1]',
            content
        )
        
        # 4. Update additional_prompt conditions
        content = re.sub(
            r'if not isinstance\(self\.base_model, OllamaModel\):',
            'if not isinstance(self.base_model, (OllamaModel, AWSBedrockModel)):',
            content
        )
        
        # Alternative pattern for additional_prompt
        content = re.sub(
            r'if not isinstance\(self\.base_model, OllamaModel\)',
            'if not isinstance(self.base_model, (OllamaModel, AWSBedrockModel))',
            content
        )
        
        # 5. Add AWS Bedrock model initialization in main function
        # Find the ollama elif block and add aws_bedrock after it
        ollama_pattern = r'(elif base_lm == ["\']ollama["\']:\s*\n(?:.*\n)*?.*OllamaModel[^\n]*\n)'
        ollama_match = re.search(ollama_pattern, content)
        
        if ollama_match and 'elif base_lm == "aws_bedrock"' not in content:
            ollama_block = ollama_match.group(1)
            aws_bedrock_block = '''    elif base_lm == "aws_bedrock":
        base_model = AWSBedrockModel(
            model_id=model_name or "meta.llama3-1-8b-instruct-v1:0",
            aws_region=aws_region,
            bearer_token=bearer_token,
            additional_prompt="ANSWER"
        )
'''
            content = content.replace(ollama_block, ollama_block + aws_bedrock_block)
        
        # 6. Add aws_region and bearer_token parameters to main function
        # Find the main function definition and add the new parameters
        main_pattern = r'def main\([^)]*\):'
        main_match = re.search(main_pattern, content)
        
        if main_match:
            main_def = main_match.group(0)
            if 'aws_region' not in main_def and 'bearer_token' not in main_def:
                # Add parameters before the closing )
                new_main_def = main_def[:-2] + ',\n    aws_region="us-east-1",\n    bearer_token=None\n):'
                content = content.replace(main_def, new_main_def)
        
        # 7. Handle special cases for IO files that might not have additional_prompt
        if '/IO/' in file_path and 'additional_prompt="NONE"' in content:
            content = re.sub(
                r'base_model = AWSBedrockModel\(\s*model_id=model_name[^)]*additional_prompt="ANSWER"[^)]*\)',
                '''base_model = AWSBedrockModel(
            model_id=model_name or "meta.llama3-1-8b-instruct-v1:0",
            aws_region=aws_region,
            bearer_token=bearer_token,
            additional_prompt="NONE"
        )''',
                content
            )
        
        # Only write if content changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
        
    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        return False

def main():
    """Update all inference.py and cot_inference.py files."""
    
    base_dir = Path("c:/Users/moiso/Documents/neurometric-ai/benchmark-mvp/tools/Sys2Bench/methods")
    
    # Find all inference.py and cot_inference.py files
    inference_files = []
    for pattern in ["**/inference.py", "**/cot_inference.py"]:
        inference_files.extend(base_dir.glob(pattern))
    
    # Remove duplicates
    inference_files = list(set(inference_files))
    
    print(f"Found {len(inference_files)} inference files to update")
    
    updated_count = 0
    for file_path in inference_files:
        print(f"Updating {file_path}")
        if update_inference_file(file_path):
            updated_count += 1
            print(f"  ✓ Updated")
        else:
            print(f"  - No changes needed")
    
    print(f"\nCompleted: {updated_count} files updated out of {len(inference_files)} total")

if __name__ == "__main__":
    main()
