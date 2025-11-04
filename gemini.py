import os, json, random
from tt_datasets import load_tt_datasets
from google import genai
from google.genai import types
from col_tt_datasets import push_to_hub
from tqdm import tqdm

def generate_response(instruction: str, row: dict, model: str, gemini_api_key: str):
    client = genai.Client(api_key=gemini_api_key)
    response = client.models.generate_content(
        model=model, 
        contents=row.to_dict(), 
        config=types.GenerateContentConfig(system_instruction=instruction)
    )
    return response.text

if __name__ == "__main__":
    api_path = "api.json"
    with open(api_path, "r") as f:
        api_keys = json.load(f)
    gemini_api_key = api_keys[0]["gemini_api_key"]
    hf_token = api_keys[0]["hf_token"]
    model = "gemini-2.5-flash"

    prompt_path = "llm_prompts.json"
    with open(prompt_path, "r") as f:
        gemini_prompts = json.load(f)["gemini_prompts"]

    seed = 1126
    ds = load_tt_datasets("Oscarshih/ee599-tt-datasets", seed)
    print(ds.column_names)

    qa_pairs = []
    for i, row in tqdm(enumerate(ds['train'])):
        instruction = gemini_prompts
        response = generate_response(instruction, row, model, gemini_api_key)
        qa_pairs.append({
            "question": row['question'],
            "reference_answer": row['answer'],
            "reference_reasoning": row['reasoning'] if row['reasoning'] else "None",
            "generated_answer": response.split("Justification: ")[0].strip(),
            "generated_reasoning": response.split("Justification: ")[1].strip()
        })
    
    out_path = "gemini_qa_pairs.jsonl"
    with open(out_path, "w") as f:
        for qa_pair in qa_pairs:
            f.write(json.dumps(qa_pair, ensure_ascii=False) + "\n")
    push_to_hub("Oscarshih/ee599-gemini-qa-pairs", filename=out_path, private=True, token=hf_token)
    