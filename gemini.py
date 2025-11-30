import os, json, random, time, io
import scipy.io.wavfile
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from preprocessing import load_tt_datasets, push_to_hub, load_audio_datasets, load_image_datasets, load_video_datasets
from google import genai
from google.genai import types
from google.genai.errors import ServerError
from tqdm import tqdm
from pydantic import BaseModel

class QAOutput(BaseModel):
    Answer: str
    Justification: str

def with_retry(call, max_retries=6, base=1.0):
    for i in range(max_retries):
        try:
            return call()
        except ServerError as e:
            if "UNAVAILABLE" not in str(e) and "429" not in str(e):
                raise
            sleep = base * (2 ** i) + random.uniform(0, 0.5)
            time.sleep(sleep)
        except Exception as e:
             # Handle 429 specifically if wrapped in other exceptions or if ServerError is not catching it
            if "429" in str(e) or "Resource exhausted" in str(e):
                sleep = base * (2 ** i) + random.uniform(0, 0.5)
                time.sleep(sleep)
                continue
            raise
    raise RuntimeError("Gemini UNAVAILABLE after retries")

def clean_json_response(text):
    try:
        cleaned_text = text.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        elif cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
        return cleaned_text.strip()
    except:
        return text

def parse_response(response_text):
    cleaned_text = clean_json_response(response_text)
    try:
        parsed = json.loads(cleaned_text)
        return parsed.get("Answer", ""), parsed.get("Justification", "")
    except Exception:
        return response_text, "JSON Parse Error"

def format_full_question(row):
    question_text = row.get('question', '')
    
    parts = [question_text]
    
    if row.get("context") is not None:
        parts.append("Context: " + str(row.get("context", "")))
        
    if row.get("constraints") is not None:
        parts.append("Constraints: " + str(row.get("constraints", "")))
        
    if row.get("options") is not None:
        options = row.get("options")
        if isinstance(options, list):
             parts.append("Options: " + "\n".join(options))
        else:
             parts.append("Options: " + str(options))
             
    # Image datasets might have 'choices' instead of 'options'
    if row.get("choices") is not None:
        choices = row.get("choices")
        if isinstance(choices, list):
             parts.append("Options: " + "\n".join(choices))
        else:
             parts.append("Options: " + str(choices))

    return "\n\n".join(parts)

def generate_text_response(instruction: str, row: dict, model: str, client: genai.Client):
    class _SafeDict(dict):
        def __missing__(self, key):
            return '{' + key + '}'
            
    # Construct prompt with all available info
    prompt = instruction.format_map(_SafeDict(
        question=row.get('question', ''),
        task=row.get('task', '')
    ))
    
    if row.get("context") is not None:
        prompt += "\n Context: " + str(row.get("context", ""))
        
    if row.get("constraints") is not None:
        prompt += "\n Constraints: " + str(row.get("constraints", ""))
        
    if row.get("options") is not None:
        options = row.get("options")
        if isinstance(options, list):
             prompt += "\n Options: " + "\n".join(options)
        else:
             prompt += "\n Options: " + str(options)

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=QAOutput
    )
    
    response = with_retry(lambda: client.models.generate_content(
        model=model,
        contents=[{"role": "user", "parts": [{"text": prompt}]}],
        config=config
    ))
    return response.text or ""

def generate_audio_response(instruction: str, row: dict, model: str, client: genai.Client):
    prompt = instruction.format(question=row.get('question', ''))
    
    audio_data = row['audio']['array']
    sampling_rate = row['audio']['sampling_rate']
    
    # Convert audio to bytes (WAV)
    with io.BytesIO() as bytes_io:
        scipy.io.wavfile.write(bytes_io, sampling_rate, audio_data)
        audio_bytes = bytes_io.getvalue()
    
    part = types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav")
    
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=QAOutput
    )
    
    response = with_retry(lambda: client.models.generate_content(
        model=model,
        contents=[{"role": "user", "parts": [part, {"text": prompt}]}],
        config=config
    ))
    return response.text or ""

def generate_image_response(instruction: str, row: dict, model: str, client: genai.Client):
    prompt = instruction.format(question=row.get('question', ''))
    
    image = row['image'] # PIL Image
    
    # Check if options exist for image (some VQA has choices)
    if row.get("choices"):
         prompt += "\n Options: " + str(row.get("choices"))

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=QAOutput
    )

    # Convert PIL image to bytes if needed or pass directly if SDK supports.
    # google-genai SDK often supports PIL Image directly, but converting to bytes is safer for compatibility.
    with io.BytesIO() as bytes_io:
        image.save(bytes_io, format="PNG")
        image_bytes = bytes_io.getvalue()

    part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
    
    response = with_retry(lambda: client.models.generate_content(
        model=model,
        contents=[{"role": "user", "parts": [part, {"text": prompt}]}],
        config=config
    ))
    return response.text or ""

def generate_video_response(instruction: str, row: dict, model: str, client: genai.Client, hf_token: str = None):
    prompt = instruction.format(question=row.get('question', ''))
    
    video_rel_path = row.get('video_path')
    # Download or find video file
    # Assuming USCECE/Video_datasets structure
    video_path = hf_hub_download(repo_id="USCECE/Video_datasets", repo_type="dataset", filename=video_rel_path, token=hf_token)
    
    # Upload file
    video_file = client.files.upload(path=video_path)
    
    # Wait for processing
    while video_file.state.name == "PROCESSING":
        time.sleep(2)
        video_file = client.files.get(name=video_file.name)
        
    if video_file.state.name == "FAILED":
        raise ValueError(f"Video processing failed for {video_rel_path}")

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=QAOutput
    )

    response = with_retry(lambda: client.models.generate_content(
        model=model,
        contents=[{"role": "user", "parts": [video_file, {"text": prompt}]}],
        config=config
    ))
    return response.text or ""

def get_processed_ids(output_file):
    processed = {}
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    modality = record.get("modality")
                    rid = record.get("id")
                    if modality and rid is not None:
                        if modality not in processed:
                            processed[modality] = set()
                        processed[modality].add(rid)
                except json.JSONDecodeError:
                    continue
    return processed

def process_dataset(dataset_name, dataset_type, prompts, client, model, output_file, hf_token, processed_ids=None):
    print(f"Processing {dataset_name} ({dataset_type})...")
    
    if dataset_type == "text":
        ds = load_tt_datasets(dataset_name)
    elif dataset_type == "audio":
        ds = load_audio_datasets(dataset_name, truncate_duration=30, truncated=False)
    elif dataset_type == "image":
        ds = load_image_datasets(dataset_name, resized=False, resized_size=(0,0))
    elif dataset_type == "video":
        ds = load_video_datasets(dataset_name)
    else:
        return

    # Determine prompts key
    if dataset_type == "text":
        instruction = prompts["gemini_prompts"]
    elif dataset_type == "audio":
        instruction = prompts["gemini_audio_prompts"]
    elif dataset_type == "image":
        instruction = prompts["gemini_image_prompts"]
    elif dataset_type == "video":
        instruction = prompts["gemini_video_prompts"]

    if processed_ids is None:
        processed_ids = set()

    # Check if file exists to resume? (Simplification: append mode)
    
    for i, row in tqdm(enumerate(ds['train']), total=len(ds['train'])):
        if i in processed_ids:
            continue
            
        try:
            if dataset_type == "text":
                response_text = generate_text_response(instruction, row, model, client)
                ref_ans = row['answer']
                ref_reasoning = row.get('reasoning') or "None"
            elif dataset_type == "audio":
                response_text = generate_audio_response(instruction, row, model, client)
                ref_ans = row.get('caption') or row.get('answer') # specific to USCECE/Audio_datasets
                ref_reasoning = "None"
            elif dataset_type == "image":
                response_text = generate_image_response(instruction, row, model, client)
                ref_ans = row['answer']
                ref_reasoning = "None"
            elif dataset_type == "video":
                response_text = generate_video_response(instruction, row, model, client, hf_token)
                ref_ans = row['answer']
                ref_reasoning = "None"
                
            generated_answer, generated_reasoning = parse_response(response_text)
            
            response_dict = {
                "id": i,
                "question": format_full_question(row),
                "reference_answer": ref_ans,
                "reference_reasoning": ref_reasoning,
                "generated_answer": generated_answer,
                "generated_reasoning": generated_reasoning,
                "modality": dataset_type
            }
            
            with open(output_file, "a") as f:
                f.write(json.dumps(response_dict, ensure_ascii=False) + "\n")
                
        except Exception as e:
            print(f"Error processing row {i}: {e}")
            continue

if __name__ == "__main__":
    api_path = "api.json"
    with open(api_path, "r") as f:
        api_keys = json.load(f)
    gemini_api_key = api_keys[0]["gemini_token"]
    hf_token = api_keys[0]["hf_token"]
    model = "gemini-2.5-flash"

    client = genai.Client(api_key=gemini_api_key)

    prompt_path = "llm_prompts.json"
    with open(prompt_path, "r") as f:
        prompts = json.load(f)

    # Define tasks
    tasks = [
        ("USCECE/filtered-tt-datasets", "text"),
        ("USCECE/Audio_datasets", "audio"),
        ("USCECE/Image_datasets", "image"),
        ("USCECE/Video_datasets", "video")
    ]
    
    output_file = "gemini_multimodal_qa_pairs.jsonl"
    
    processed_progress = get_processed_ids(output_file)
    
    for ds_name, ds_type in tasks:
        existing_ids = processed_progress.get(ds_type, set())
        process_dataset(ds_name, ds_type, prompts, client, model, output_file, hf_token, processed_ids=existing_ids)

    push_to_hub("USCECE/gemini-multimodal-QApairs", filename=output_file, private=False, token=hf_token)
