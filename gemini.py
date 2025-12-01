import os, json, random, time, io
import scipy.io.wavfile
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download, HfApi
from preprocessing import load_tt_datasets, push_to_hub, load_audio_datasets, load_image_datasets, load_video_datasets
from google import genai
from google.genai import types
from google.genai.errors import ServerError
from tqdm import tqdm
from pydantic import BaseModel
from datasets import Dataset, Audio, Image, Video

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
    
    audio_entry = row['audio']
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
        raise ValueError(f"Audio data missing or invalid format: {audio_entry.keys()}")
    
    part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
    
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
    image = row['image']
    if row.get("choices"):
         prompt += "\n Options: " + str(row.get("choices"))

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=QAOutput
    )

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
    video_path = hf_hub_download(repo_id="USCECE/Video_datasets", repo_type="dataset", filename=video_rel_path, token=hf_token)
    video_file = client.files.upload(file=video_path)
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
        contents=[video_file, prompt],
        config=config
    ))
    return response.text or ""

def process_dataset(dataset_name, dataset_type, prompts, client, model, hf_token, processed_ids=None):
    print(f"Processing {dataset_name} ({dataset_type})...")
    
    if dataset_type == "text":
        ds = load_tt_datasets(dataset_name)
        output_file = "gemini_tt_qa_pairs.jsonl"
    elif dataset_type == "audio":
        ds = load_audio_datasets(dataset_name, truncate_duration=30, truncated=False)
        output_file = "gemini_audio_qa_pairs.jsonl"
    elif dataset_type == "image":
        ds = load_image_datasets(dataset_name, resized=False, resized_size=(0,0))
        output_file = "gemini_image_qa_pairs.jsonl"
    elif dataset_type == "video":
        ds = load_video_datasets(dataset_name)
        ds = ds.cast_column("video", Video(decode=False))
        output_file = "gemini_video_qa_pairs.jsonl"
    else:
        return

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
                ref_ans = row.get('caption') or row.get('answer')
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

def upload_results_as_dataset(dataset_name, dataset_type, jsonl_file, repo_name, hf_token):
    if not os.path.exists(jsonl_file):
        print(f"File {jsonl_file} not found, skipping upload.")
        return

    print(f"Preparing dataset for upload: {repo_name}...")
    
    with open(jsonl_file, "r") as f:
        results = [json.loads(line) for line in f if line.strip()]
    
    if not results:
        print("No results to upload.")
        return

    if dataset_type == "text":
        ds_out = Dataset.from_list(results)
    else:
        if dataset_type == "audio":
             src_ds = load_audio_datasets(dataset_name, truncate_duration=30, truncated=False)
             media_col = "audio"
        elif dataset_type == "image":
             src_ds = load_image_datasets(dataset_name, resized=False, resized_size=(0,0))
             media_col = "image"
        elif dataset_type == "video":
             src_ds = load_video_datasets(dataset_name)
             src_ds = src_ds.cast_column("video", Video(decode=False))
             media_col = "video"

        data_to_push = []
        api = HfApi(token=hf_token)
        api.create_repo(repo_id=repo_name, repo_type="dataset", exist_ok=True, private=False)

        for res in results:
            idx = res['id']
            item = res.copy()
            
            if idx < len(src_ds['train']):
                src_item = src_ds['train'][idx]
                
                if dataset_type == "video":
                    video_info = src_item[media_col]
                    if video_info and 'path' in video_info:
                        local_path = video_info['path']
                        filename = os.path.basename(local_path)
                        target_path = f"videos/{filename}"
                        print(f"Uploading video {filename} to {repo_name}...")
                        api.upload_file(
                            path_or_fileobj=local_path,
                            path_in_repo=target_path,
                            repo_id=repo_name,
                            repo_type="dataset"
                        )
                        item[media_col] = target_path
                else:
                    item[media_col] = src_item[media_col]
            
            data_to_push.append(item)
            
        ds_out = Dataset.from_list(data_to_push)
    
        if dataset_type == "audio":
            ds_out = ds_out.cast_column("audio", Audio())
        elif dataset_type == "image":
            ds_out = ds_out.cast_column("image", Image())
 
    print(f"Pushing {repo_name} to Hub...")
    
    if dataset_type == "video":
        updated_jsonl_name = os.path.basename(jsonl_file)
        
        with open(updated_jsonl_name, "w") as f:
            for item in data_to_push:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                
        print(f"Uploading updated {updated_jsonl_name} to {repo_name}...")
        api.upload_file(
            path_or_fileobj=updated_jsonl_name,
            path_in_repo=updated_jsonl_name,
            repo_id=repo_name,
            repo_type="dataset"
        )
        print(f"Pushed {repo_name} (manual upload)")
    else:
        ds_out.push_to_hub(repo_name, token=hf_token, private=False)
        print(f"Pushed {repo_name}")

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

    tasks = [
        ("USCECE/Text_datasets", "text"),
        ("USCECE/Audio_datasets", "audio"),
        ("USCECE/Image_datasets", "image"),
        ("USCECE/Video_datasets", "video")
    ]
    
    for ds_name, ds_type in tasks:
        if ds_type == "text":
            output_file = "gemini_text_qa_pairs.jsonl"
        elif ds_type == "audio":
            output_file = "gemini_audio_qa_pairs.jsonl"
        elif ds_type == "image":
            output_file = "gemini_image_qa_pairs.jsonl"
        elif ds_type == "video":
            output_file = "gemini_video_qa_pairs.jsonl"
            
        processed_ids = get_processed_ids(output_file).get(ds_type, set())
        process_dataset(ds_name, ds_type, prompts, client, model, hf_token, processed_ids=processed_ids)

    upload_results_as_dataset("USCECE/Text_datasets", "text", "gemini_text_qa_pairs.jsonl", "USCECE/gemini-text-qa-pairs", hf_token)
    upload_results_as_dataset("USCECE/Audio_datasets", "audio", "gemini_audio_qa_pairs.jsonl", "USCECE/gemini-audio-qa-pairs", hf_token)
    upload_results_as_dataset("USCECE/Image_datasets", "image", "gemini_image_qa_pairs.jsonl", "USCECE/gemini-image-qa-pairs", hf_token)
    upload_results_as_dataset("USCECE/Video_datasets", "video", "gemini_video_qa_pairs.jsonl", "USCECE/gemini-video-qa-pairs", hf_token)
