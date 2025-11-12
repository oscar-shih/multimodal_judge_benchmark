import os, json, random
from preprocessing import load_tt_datasets, push_to_hub, load_audio_datasets, load_image_datasets, load_video_datasets
from google import genai
from google.genai import types
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

    tt_ds = load_tt_datasets("USCECE/tt-datasets")
    tt_qa_pairs = []
    for i, row in tqdm(enumerate(tt_ds['train'])):
        instruction = gemini_prompts
        response = generate_response(instruction, row, model, gemini_api_key)
        tt_qa_pairs.append({
            "id": i,
            "question": row['question'],
            "reference_answer": row['answer'],
            "reference_reasoning": row['reasoning'] if row['reasoning'] else "None",
            "generated_answer": response.split("Justification: ")[0].strip(),
            "generated_reasoning": response.split("Justification: ")[1].strip()
        })
    
    out_path = "gemini_tt_qa_pairs.jsonl"
    with open(out_path, "w") as f:
        for qa_pair in tt_qa_pairs:
            f.write(json.dumps(qa_pair, ensure_ascii=False) + "\n")
    push_to_hub("USCECE/gemini-tt-qa-pairs", filename=out_path, private=True, token=hf_token)

    # TODO: Process the audio, image, and video datasets and generate the qa pairs for them.
    truncate_duration = 30.0 # TODO: Determine the longest truncation length for Gemini.
    truncated = True
    audio_ds = load_audio_datasets("USCECE/Audio_datasets", truncate_duration, truncated)
    audio_qa_pairs = []
    for i, row in tqdm(enumerate(audio_ds['train'])):
        instruction = gemini_prompts
        response = generate_response(instruction, row, model, gemini_api_key)
        audio_qa_pairs.append({
            "id": i,
            "audio": row['audio'],
            "question": row['question'],
            "reference_answer": row['caption'] if row['caption'] else "None",
            "generated_answer": response.split("Justification: ")[0].strip(),
            "generated_reasoning": response.split("Justification: ")[1].strip()
        })
    out_path = "gemini_audio_qa_pairs.jsonl"
    with open(out_path, "w") as f:
        for qa_pair in audio_qa_pairs:
            f.write(json.dumps(qa_pair, ensure_ascii=False) + "\n")
    push_to_hub("USCECE/gemini-audio-qa-pairs", filename=out_path, private=True, token=hf_token)

    # TODO: Process the image datasets and generate the qa pairs for them.
    resized = True
    resized_size = (224, 224) # TODO: Determine the bestresized size for Gemini.
    image_ds = load_image_datasets("USCECE/Image-Dataset-FP", resized, resized_size)
    image_qa_pairs = []
    for i, row in tqdm(enumerate(image_ds['train'])):
        instruction = gemini_prompts
        response = generate_response(instruction, row, model, gemini_api_key)
        image_qa_pairs.append({
            "id": i,
            "image": row['image'],
            "question": row['question'] + row['choices'],
            "reference_answer": row['answer'] if row['answer'] else "None",
            "generated_answer": response.split("Justification: ")[0].strip(),
            "generated_reasoning": response.split("Justification: ")[1].strip()
        })
    out_path = "gemini_image_qa_pairs.jsonl"
    with open(out_path, "w") as f:
        for qa_pair in image_qa_pairs:
            f.write(json.dumps(qa_pair, ensure_ascii=False) + "\n")
    push_to_hub("USCECE/gemini-image-qa-pairs", filename=out_path, private=True, token=hf_token)

    # TODO: Process the video datasets and generate the qa pairs for them.
    video_ds = load_video_datasets("USCECE/Video_datasets")
    video_qa_pairs = []
    for i, row in tqdm(enumerate(video_ds['train'])):
        instruction = gemini_prompts
        response = generate_response(instruction, row, model, gemini_api_key)
        video_qa_pairs.append({
            "id": i,
            "video": row['video'],
            "question": row['question'],
            "reference_answer": row['caption'] if row['caption'] else "None",
            "generated_answer": response.split("Justification: ")[0].strip(),
            "generated_reasoning": response.split("Justification: ")[1].strip()
        })
    out_path = "gemini_video_qa_pairs.jsonl"
    with open(out_path, "w") as f:
        for qa_pair in video_qa_pairs:
            f.write(json.dumps(qa_pair, ensure_ascii=False) + "\n")
    push_to_hub("USCECE/gemini-video-qa-pairs", filename=out_path, private=True, token=hf_token)