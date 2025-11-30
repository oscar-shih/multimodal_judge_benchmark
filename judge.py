import os
import json
from openai import OpenAI
from tqdm import tqdm
from pathlib import Path
import argparse

def generate_judgement_gpt( 
    input_metadata_path: Path,
    judge_prompts_path: Path,
    api_key_path: Path,
    output__path: Path
):
    judge_prompts = json.load(judge_prompts_path.open(mode='r'))
    with open(api_key_path, "r") as f:
        api_keys = json.load(f)
    client = OpenAI(api_key=api_keys[0]["gpt5_token"])
    metadata = [json.loads(line) for line in input_metadata_path.open(mode='r').readlines()]
    for qa_pair in tqdm(metadata):
        with open(output_path, "a") as f:
            
            response = client.chat.completions.create(
                model='gpt-5',
                messages=[
                    {
                        'role': 'system',
                        'content': judge_prompts,
                    },
                    {
                        'role': 'user',
                        'content': qa_pair
                    }
                ]
            )
            judgement = response.choices[0].message.content
            f.write(json.dumps(judgement) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_metadata_path", type=Path)
    parser.add_argument("--judge_prompts_path", type=Path)
    parser.add_argument("--api_key_path", type=Path)
    parser.add_argument("--output_judgement_path", type=Path)
    generate_judgement_gpt(**vars(parser.parse_args()))