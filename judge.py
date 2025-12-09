import os, json, random, time, io
import scipy.io.wavfile
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from preprocessing import load_tt_datasets, load_audio_datasets, load_image_datasets, load_video_datasets
from google import genai
from google.genai import types
from google.genai.errors import ServerError
from tqdm import tqdm
from pydantic import BaseModel
from datasets import Video, load_dataset

class JudgeOutput(BaseModel):
    Score: int
    Judgement: str
    Error: str

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
        return parsed.get("Score", -1), parsed.get("Judgement", ""), parsed.get("Error", "None")
    except Exception:
        return -1, response_text, "JSON Parse Error"

def construct_judge_prompt(instruction, row_data):
    prompt = instruction + "\n\n"
    prompt += f"Question: {row_data.get('question', '')}\n"
    prompt += f"Reference Answer: {row_data.get('reference_answer', '')}\n"
    prompt += f"Reference Reasoning: {row_data.get('reference_reasoning', 'None')}\n"
    prompt += f"Model Answer: {row_data.get('generated_answer', '')}\n"
    prompt += f"Model Reasoning: {row_data.get('generated_reasoning', '')}\n"
    return prompt

def evaluate_text(instruction, row_data, model, client):
    prompt = construct_judge_prompt(instruction, row_data)
    
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=JudgeOutput
    )
    
    response = with_retry(lambda: client.models.generate_content(
        model=model,
        contents=[{"role": "user", "parts": [{"text": prompt}]}],
        config=config
    ))
    return response.text or ""

def evaluate_audio(instruction, row_data, original_row, model, client):
    prompt = construct_judge_prompt(instruction, row_data)
    
    audio_entry = original_row['audio']
    mime_type = "audio/wav"
    
    if 'array' in audio_entry and audio_entry['array'] is not None:
        audio_data = audio_entry['array']
        sampling_rate = audio_entry['sampling_rate']
        with io.BytesIO() as bytes_io:
            scipy.io.wavfile.write(bytes_io, sampling_rate, audio_data)
            audio_bytes = bytes_io.getvalue()
    elif 'bytes' in audio_entry and audio_entry['bytes'] is not None:
        audio_bytes = audio_entry['bytes']
        if audio_bytes.startswith(b'RIFF'):
            mime_type = "audio/wav"
        elif audio_bytes.startswith(b'\xff\xfb') or audio_bytes.startswith(b'\xff\xf3') or audio_bytes.startswith(b'\xff\xf2') or audio_bytes.startswith(b'ID3'):
            mime_type = "audio/mp3"
    else:
        raise ValueError(f"Audio data missing or invalid format")
    
    part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
    
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=JudgeOutput
    )
    
    response = with_retry(lambda: client.models.generate_content(
        model=model,
        contents=[{"role": "user", "parts": [part, {"text": prompt}]}],
        config=config
    ))
    return response.text or ""

def evaluate_image(instruction, row_data, original_row, model, client):
    prompt = construct_judge_prompt(instruction, row_data)
    image = original_row['image']
    
    with io.BytesIO() as bytes_io:
        image.save(bytes_io, format="PNG")
        image_bytes = bytes_io.getvalue()

    part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
    
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=JudgeOutput
    )
    
    response = with_retry(lambda: client.models.generate_content(
        model=model,
        contents=[{"role": "user", "parts": [part, {"text": prompt}]}],
        config=config
    ))
    return response.text or ""

def evaluate_video(instruction, row_data, original_row, model, client, hf_token=None, repo_id="USCECE/Video_datasets"):
    prompt = construct_judge_prompt(instruction, row_data)
    video_rel_path = original_row.get('video_path')
    
    if not video_rel_path:
        if 'video' in original_row:
            if isinstance(original_row['video'], dict):
                video_rel_path = original_row['video'].get('path')
            elif isinstance(original_row['video'], str):
                video_rel_path = original_row['video']
    
    if video_rel_path and video_rel_path.startswith("./"):
        video_rel_path = video_rel_path[2:]
        
    video_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=video_rel_path, token=hf_token)
    video_file = client.files.upload(file=video_path)
    while video_file.state.name == "PROCESSING":
        time.sleep(2)
        video_file = client.files.get(name=video_file.name)
        
    if video_file.state.name == "FAILED":
        raise ValueError(f"Video processing failed for {video_rel_path}")

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=JudgeOutput
    )

    response = with_retry(lambda: client.models.generate_content(
        model=model,
        contents=[video_file, prompt],
        config=config
    ))
    return response.text or ""

def process_judgement(dataset_name, dataset_type, prompts, client, model, hf_token, generated_file, target_ids=None):
    with open(generated_file, "r") as f:
        generated_results = [json.loads(line) for line in f if line.strip()]
    
    gen_map = {res['id']: res for res in generated_results}
    
    if dataset_type == "text":
        ds = load_tt_datasets(dataset_name)
    elif dataset_type == "audio":
        ds = load_audio_datasets(dataset_name, truncate_duration=30, truncated=False)
    elif dataset_type == "image":
        ds = load_image_datasets(dataset_name, resized=False, resized_size=(0,0))
    elif dataset_type == "video":
        ds = load_video_datasets(dataset_name)
        ds = ds.cast_column("video", Video(decode=False))
    else:
        return

    instruction = prompts["judge_prompts"]
    output_file = generated_file.replace(".jsonl", "_results.jsonl")
    
    processed_ids = set()
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if "id" in rec:
                        processed_ids.add(rec["id"])
                except:
                    pass
    
    for res in tqdm(generated_results):
        rid = res['id']
        if target_ids is not None and rid not in target_ids:
            continue
        if rid in processed_ids:
            continue
        
        try:
            original_row = ds['train'][rid]
        except IndexError:
            print(f"Index {rid} out of bounds for dataset {dataset_name}")
            continue
            
        try:
            if dataset_type == "text":
                response_text = evaluate_text(instruction, res, model, client)
            elif dataset_type == "audio":
                response_text = evaluate_audio(instruction, res, original_row, model, client)
            elif dataset_type == "image":
                response_text = evaluate_image(instruction, res, original_row, model, client)
            elif dataset_type == "video":
                # Use the main Video_datasets repo for the video files, assuming filenames match
                response_text = evaluate_video(instruction, res, original_row, model, client, hf_token, repo_id="USCECE/Video_datasets")
            
            score, judgement, error = parse_response(response_text)
            
            result_dict = res.copy()
            result_dict.update({
                "judge_score": score,
                "judge_reason": judgement,
                "judge_error": error,
                "judge_model": model
            })
            
            with open(output_file, "a") as f:
                f.write(json.dumps(result_dict, ensure_ascii=False) + "\n")
                
        except Exception as e:
            print(f"Error judging row {rid}: {e}")
            continue

if __name__ == "__main__":
    api_path = "api.json"
    with open(api_path, "r") as f:
        api_keys = json.load(f)
    gemini_api_key = api_keys[0]["gemini_token"]
    hf_token = api_keys[0]["hf_token"]
    model = "gemini-2.5-flash" 

    client = genai.Client(api_key=gemini_api_key)

    prompt_path = "judge_prompts.json"
    with open(prompt_path, "r") as f:
        prompts = json.load(f)

    tasks = [
        ("USCECE/phi4_tt", "text", "phi4_text_qa_pairs.jsonl", None),
        ("USCECE/phi4_video", "video", "phi4_video_qa_pairs.jsonl", None),
        ("USCECE/phi4_image", "image", "phi4_image_qa_pairs.jsonl", None),
        ("USCECE/phi4_audio", "audio", "phi4_audio_qa_pairs.jsonl", None),
    ]
    
    for ds_name, ds_type, gen_file, target_ids in tasks:
        if not os.path.exists(gen_file):
            print(f"Generating {gen_file} from {ds_name}...")
            try:
                ds = load_dataset(ds_name, split="train")
                with open(gen_file, "w") as f:
                    for i, row in enumerate(ds):
                        # Use index as ID if not present or to be safe
                        clean_row = {
                            "id": i, 
                            "question": row.get("question"),
                            "reference_answer": row.get("reference_answer"),
                            "reference_reasoning": row.get("reference_reasoning"),
                            "generated_answer": row.get("generated_answer"),
                            "generated_reasoning": row.get("generated_reasoning")
                        }
                        f.write(json.dumps(clean_row, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"Failed to generate {gen_file}: {e}")
                continue
                    
        process_judgement(ds_name, ds_type, prompts, client, model, hf_token, gen_file, target_ids=target_ids)
