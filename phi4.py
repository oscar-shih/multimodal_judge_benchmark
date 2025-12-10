import os
import io
import argparse

import torch
import numpy as np
import pandas as pd

from datasets import load_dataset, Audio, Video, Dataset
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import imageio.v2 as imageio
import soundfile as sf
from huggingface_hub import HfApi, upload_file

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print("Using device:", device)

model_id = "microsoft/Phi-4-multimodal-instruct"

print("Loading processor & model...")
processor = AutoProcessor.from_pretrained(
    model_id,
    trust_remote_code=True
)

dtype = torch.bfloat16 if device == "cuda" else "auto"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map=device,
    torch_dtype=dtype,
    trust_remote_code=True,
    _attn_implementation="eager",
)
model.eval()

DATA_ROOT = os.getenv("DATA_ROOT", "data")

# =========================================================
# Audio
# =========================================================

GLOBAL_POLICY_PROMPT_AUDIO = """Global Prompt Policy for Audio-Language Tasks
You are a careful, concise model for audio-language reasoning. Use ONLY the provided
audio clip and text input (e.g., question) as evidence. Ignore prior knowledge and external
information. When audio cues conflict with textual descriptions, prefer the audio evidence.
For each task, you must:
- Carefully interpret sounds, voices, and background cues.
- Integrate audio perception with textual understanding when applicable.
- Avoid assumptions beyond what can be heard or explicitly stated.
At the end, output EXACTLY two lines:
Justification: <Short explanation based on audible cues or evidence>
Final Answer: <...>
No extra text before or after these two lines.
"""

QUESTION_ANSWER_PROMPT = """Question-Answer Prompt
[Task = {task_family}]
Please answer the following question based on the provided audio clip.
Use ONLY the sounds in the audio and the accompanying text as evidence.
Audio: <|audio_1|>
Question: {question}
Rules:
- Carefully listen to the audio to identify relevant sounds, voices, or background events.
- Base your response solely on what can be heard or clearly implied from the audio.
- If no relevant sound is present, describe what is actually audible instead.
Justification: <Short explanation based on audible cues or reasoning>
Final Answer: <Concise answer or caption>
"""

INSTRUCTION_FOLLOWING_PROMPT_AUDIO = """Instruction-Following Prompt
[Task = {task_family}]
Given the audio clip and the provided instruction, follow the instruction carefully and
respond accordingly.
Use ONLY the information from the audio and the given text input as evidence.
Audio: <|audio_1|>
Instruction: {question}
Rules:
- Listen carefully to the audio and identify cues related to the instruction.
- Follow the instruction precisely.
- Base your output only on what can be heard in the audio clip.
Justification: <Brief reason or evidence from the audio supporting the output>
Final Answer: <Concise completion or response>
"""

def run_audio_pipeline():
    print("\n=== [AUDIO] Loading dataset multi-judge/Audio_datasets ===")
    ds = load_dataset("multi-judge/Audio_datasets", split="train")

    ds = ds.cast_column("audio", Audio(decode=False))

    num_samples = len(ds)
    print(f"Total samples in dataset: {num_samples}")

    user_prompt = "<|user|>"
    assistant_prompt = "<|assistant|>"
    end_prompt = "<|end|>"

    records = []

    for idx in range(num_samples):
        sample = ds[idx]

        print("\n" + "=" * 60)
        print(f"SAMPLE {idx} / {num_samples - 1}")
        print("=" * 60)

        audio_bytes = sample["audio"]["bytes"]
        audio_file_obj = io.BytesIO(audio_bytes)
        audio_array, sampling_rate = sf.read(audio_file_obj)

        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)

        question = sample["question"]
        task_family = sample.get("task", "Question-Answer")
        caption = sample.get("caption", None)
        source = sample.get("source", None)
        sample_id = sample.get("id", idx)

        print("\n=== Sample Loaded ===")
        print("ID:", sample_id)
        print("Task:", task_family)
        print("Question:", question)
        print("Caption:", caption)
        print("Audio shape:", audio_array.shape, "| SR:", sampling_rate)

        if task_family == "Instruction-Following":
            task_prompt = INSTRUCTION_FOLLOWING_PROMPT_AUDIO.format(
                task_family=task_family,
                question=question,
            )
        else:
            task_prompt = QUESTION_ANSWER_PROMPT.format(
                task_family=task_family,
                question=question,
            )

        full_user_content = GLOBAL_POLICY_PROMPT_AUDIO + "\n\n" + task_prompt
        prompt = f"{user_prompt}{full_user_content}{end_prompt}{assistant_prompt}"

        print("\n=== PROMPT ===")
        print(prompt)

        inputs = processor(
            text=prompt,
            audios=[(audio_array, sampling_rate)],
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            generate_ids = model.generate(
                **inputs,
                max_new_tokens=1280,
            )

        generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]

        response = processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        print("\n=== MODEL RESPONSE ===")
        print(response)

        records.append({
            "id": int(sample_id),
            "task": task_family,
            "question": question,
            "caption": caption,
            "source": source,
            "phi4_response": response,
        })

    os.makedirs("out", exist_ok=True)
    parquet_path = "out/phi4_audio_responses.parquet"

    df = pd.DataFrame(records).reset_index(drop=True)
    df.to_parquet(parquet_path, index=False)

    print(f"\n[AUDIO] Saved Parquet to: {parquet_path}")
    print("DataFrame shape:", df.shape)

    dataset_out = Dataset.from_pandas(df)
    dataset_out.push_to_hub("yulianawu/phi4_audio")
    print("\n[AUDIO] Pushed dataset to HuggingFace Hub: yulianawu/phi4_audio")


# =========================================================
# Image pipeline
# =========================================================

GLOBAL_POLICY_IMAGE = """Global Prompt Policy
You are a careful and concise vision-language model. Keep detailed reasoning internal. Use
ONLY the information visible in the image and question provided (and choice if possible).
Do not hallucinate objects, text, or facts that cannot be confirmed from the image.
At the end, output EXACTLY two lines:
Justification: <One short sentence summarizing the key visual evidence>
Final Answer: <final answer>
No extra text before/after those two lines.
"""

def build_prompt_image(task, question, choices):
    choices_text = ""
    if choices:
        choices_text = "\nChoices (optional):\n" + "\n".join(choices)

    t = (task or "").strip()

    if t == "Image Captioning":
        body = f"""[Task = {t}]
You are answering an Image Captioning question. If answer choices are provided, select
exactly one choice.
Question: {question}{choices_text}
Rules:
- Use only information visible in the image.
- Keep the caption concise, factual, and contextually plausible.
- If multiple-choice options are provided, select exactly one option.
Justification: <Provide one sentence that identifies the key visual evidence>
Final Answer: <answer or choice>"""

    elif t == "Chart Understanding":
        body = f"""[Task = {t}]
You are answering a Chart Understanding question. Your answer must rely solely on the
information explicitly shown in the chart.
Question: {question}
Rules:
- Use only information visible in the chart (axes, labels, legends, values, markers).
- Report numerical values exactly as displayed.
- Do not infer or assume trends that are not explicitly shown.
Justification: <Provide one sentence describing the key chart evidence such as values,
trends, or comparisons that support the final answer>
Final Answer: <answer>"""

    elif t == "Math Reasoning":
        body = f"""[Task = {t}]
You are answering a Math Reasoning question based on the visual diagram or information
presented in the image.
Question: {question}{choices_text}
Rules:
- Use only the mathematical information visible in the diagram or the problem statement.
- Do not assume values, shapes, or relationships that are not explicitly shown.
- If multiple-choice options are provided, select exactly one option.
Justification: <Provide one sentence identifying the key mathematical elements or relationships used to obtain the answer>
Final Answer: <answer or choice>"""

    else:
        body = f"""[Task = {t}]
You are answering a visual question based only on the provided image.
Question: {question}{choices_text}
Rules:
- Use only information visible in the image.
Justification: <Provide one sentence describing the key visual evidence>
Final Answer: <answer or choice>"""

    full_text = GLOBAL_POLICY_IMAGE + "\n\n" + body
    prompt = "<|user|><|image_1|>" + full_text + "<|end|><|assistant|>"
    return prompt

def run_phi4_image(task, question, choices, image):
    prompt = build_prompt_image(task, question, choices)

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    resized_size = (224, 224)
    image = image.resize(resized_size)

    inputs = processor(
        text=prompt,
        images=[image],
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1280,
            do_sample=False,
        )

    generated = outputs[0, inputs["input_ids"].shape[1]:]
    text = processor.decode(generated, skip_special_tokens=True)
    return text.strip()

def run_image_pipeline():
    print("\n=== [IMAGE] Loading dataset multi-judge/Image_datasets ===")
    ds = load_dataset("multi-judge/Image_datasets", split="train")
    print(f"Loaded {len(ds)} image samples.")

    results = {
        "idx": [],
        "id": [],
        "task": [],
        "question": [],
        "choices": [],
        "ground_truth": [],
        "model_answer": [],
    }

    print("[IMAGE] Running inference...")

    for idx, row in enumerate(ds):
        print("=" * 70)
        print(f"[Sample {idx}]")

        task = row.get("task", "")
        question = row["question"]
        choices = row.get("choices", None)

        model_answer = run_phi4_image(
            task=task,
            question=question,
            choices=choices,
            image=row["image"],
        )

        results["idx"].append(idx)
        results["id"].append(row.get("id", ""))
        results["task"].append(task)
        results["question"].append(question)
        results["choices"].append(choices)
        results["ground_truth"].append(row["answer"])
        results["model_answer"].append(model_answer)

    output_path = "phi4_image_results.parquet"
    df = pd.DataFrame(results)
    df.to_parquet(output_path, index=False)

    print(f"\n[IMAGE] Results saved locally as: {output_path}")

    repo_id = "yulianawu/phi4_image"
    print("\n[IMAGE] Uploading parquet to HuggingFace Hub...")
    api = HfApi()

    upload_file(
        path_or_fileobj=output_path,
        path_in_repo=output_path,
        repo_id=repo_id,
        repo_type="dataset"
    )

    print(f"[IMAGE] Upload complete: https://huggingface.co/datasets/{repo_id}")


# =========================================================
# TT
# =========================================================

GLOBAL_PROMPT_POLICY_TT = """Global Prompt Policy
You are a careful, concise model. Keep detailed reasoning internal. Use ONLY the
information provided here (and optional context). If context conflicts with prior knowledge,
prefer the context.
At the end, output EXACTLY two lines:
Justification: <Some sentences or a short checklist, task-specific>
Final Answer: <...>
No extra text before/after those two lines.
"""

def build_prompt_tt(row):
    task = row["task"]
    task_family = row.get("task_family", task)
    question = row["question"]
    context = row.get("context", "")

    if task == "Expert-MCQ":
        option_A = row.get("option_A", "")
        option_B = row.get("option_B", "")
        option_C = row.get("option_C", "")
        option_D = row.get("option_D", "")

        task_prompt = f"""MCQ Prompt
[Task = {task_family}]
You are answering a multiple-choice question. If a context is provided, treat it as ground
truth. Choose exactly one option.
Question: {question}
Context (optional): {context}
Options:
A) {option_A}
B) {option_B}
C) {option_C}
D) {option_D}
Rules:
- For justifications, cite a key phrase from the context OR briefly state the elimination logic.
- Then output only the letter A, B, C, or D as the final answer.
Justification: <why this option is supported or others are ruled out>
Final Answer: <A|B|C|D>"""

    elif task == "Reading-Comprehension":
        task_prompt = f"""Extractive RC Prompt
[Task = {task_family}]
Return the SHORTEST substring of the passage that exactly answers the question. The
answer MUST be a verbatim span from the passage (case and punctuation preserved).
Passage: {context}
Question: {question}
Rules:
- The answer must be a contiguous substring of the passage.
- If multiple spans could work, return the shortest fully correct one.
- For justifications, state why you find this span to answer this question.
Justification: <quote a nearby clue or sentence index supporting the span>
Final Answer: "<exact span from passage>"""

    elif task == "Reasoning-Math":
        task_prompt = f"""Math Prompt
[Task = {task_family}]
Solve the problem and output only the final canonical answer.
Problem: {question}
Canonicalization:
- Integers: plain integer (e.g., 42).
- Fractions: lowest terms with "/" (e.g., 7/12).
- Decimals: remove trailing zeros (e.g., 3.5 not 3.5000).
- Sets/tuples: a,b,c or (1,2,3); sort elements.
- Units: include if explicitly required.
Justification: <Equations or a terse transformation leading to the result>
Final Answer: {{final answer}}"""

    elif task == "Reasoning-Code":
        task_prompt = f"""Code Reasoning Prompt
[Task = {task_family}]
Reason about code if needed, but return ONLY the required final value/string (unless the
prompt explicitly asks for code).
Prompt: {question}
Rules:
- If a numeric/string result is required, return only that value.
- Do not include code or stack traces.
Justification: <brief description of the operation/algorithm that yields the value>
Final Answer: {{final_answer}}"""

    elif task == "Instruction-Following":
        instr = row.get("prompt", question)
        constraint_list = row.get("constraint_list", "[]")

        task_prompt = f"""Instruction-Following Prompt
[Task = {task_family}]
Follow ALL constraints exactly. If constraints conflict, obey the most specific.
Instruction: {instr}
Constraints (must all pass): {constraint_list}
Rules:
- Respect formatting limits (e.g., length, JSON schema, allowed chars).
- If a safety-violating constraint appears, return a safe alternative while still satisfying
neutral formatting constraints.
Justification:
- [ ] constraint1: pass/fail + 3–6 words
- [ ] constraint2: pass/fail + 3–6 words
... (tick [x] for pass, [ ] for fail)
Final Answer: {{final_answer}}"""

    else:
        task_prompt = f"{question}"

    full_prompt = GLOBAL_PROMPT_POLICY_TT + "\n\n" + task_prompt
    return full_prompt

def run_phi4_tt(row):
    prompt = build_prompt_tt(row)
    full = f"<|user|>{prompt}<|end|><|assistant|>"

    inputs = processor(full, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1280,
            do_sample=False
        )

    generated = outputs[0, inputs["input_ids"].shape[1]:]
    text = processor.decode(generated, skip_special_tokens=True)
    return text

def run_tt_pipeline():
    print("\n=== [TT] Loading dataset multi-judge/tt_datasets ===")
    ds = load_dataset("multi-judge/tt_datasets", split="train")
    ds = ds.filter(lambda x: x["task"] != "Long-Context")
    print(f"Loaded {len(ds)} samples (filtered).")

    results = []

    for idx, row in enumerate(ds):
        print(f"\n===== [TT] Running sample {idx}/{len(ds)} =====")
        question = row["question"]

        try:
            answer = run_phi4_tt(row)
        except Exception as e:
            answer = f"ERROR: {str(e)}"

        results.append({
            "id": row["id"],
            "task": row["task"],
            "question": question,
            "answer": row["answer"],
            "source": row["source"],
            "reasoning": row["reasoning"],
            "model_answer": answer,
            "options": row["options"],
            "context": row["context"],
            "answer_type": row["answer_type"],
            "source_url": row["source_url"],
            "split": row["split"],
            "license": row["license"],
            "constraints": row["constraints"],
        })

        print("Task:", row["task"])
        print("Question:", question)
        print("Answer:", answer)

    df = pd.DataFrame(results)
    parquet_path = "phi4_tt_results.parquet"
    df.to_parquet(parquet_path)

    print(f"\n[TT] Saved results to: {parquet_path}")

    print("[TT] Pushing dataset to HuggingFace Hub...")
    hf_ds = Dataset.from_pandas(df)
    hf_ds.push_to_hub("yulianawu/phi4_tt")
    print("[TT] Upload complete: https://huggingface.co/datasets/yulianawu/phi4_tt")


# =========================================================
# Video
# =========================================================

VIDEO_QA_TEMPLATE = """Video Question Answering
You are a careful and concise vision-language model. Keep detailed reasoning internal.You
are answering a Video QA question. Answer the question ONLY based on the visual
content and events shown in the video.
Question: {question}
Rules: - Use only the information visible in the video frames.
- Consider the temporal order, motion, and interactions occurring across the clip.
- Do not assume details that are not clearly shown.
At the end, output EXACTLY two lines:
Justification: <Provide one sentence describing the key visual or temporal evidence that
supports the answer>
Final Answer: <answer>
No extra text before/after those two lines.
"""

def run_one_video(row):
    video_rel_path = row["video_path"]
    video_path = os.path.join(DATA_ROOT, video_rel_path)
    print("Video file path:", video_path)

    reader = imageio.get_reader(video_path, "ffmpeg")
    num_frames = reader.count_frames()
    print("Total frames in video:", num_frames)

    n_frames = min(8, num_frames)
    indices = np.linspace(0, num_frames - 1, n_frames, dtype=int)

    frames = []
    for idx in indices:
        frame = reader.get_data(idx)
        img = Image.fromarray(frame)
        img = img.resize((224, 224))
        frames.append(img)

    reader.close()
    print(f"Sampled {len(frames)} frames from video.")

    question = row.get(
        "question",
        "What is happening in this video clip? Describe the main events."
    )

    placeholder = "".join(f"<|image_{i+1}|>" for i in range(len(frames)))
    prompt_body = VIDEO_QA_TEMPLATE.format(question=question)
    user_content = placeholder + "\n\n" + prompt_body
    full_prompt = f"<|user|>{user_content}<|end|><|assistant|>"

    inputs = processor(
        text=full_prompt,
        images=frames,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1280,
            do_sample=False,
        )

    generated = outputs[0, inputs["input_ids"].shape[1]:]
    text = processor.decode(generated, skip_special_tokens=True)

    if device == "cuda":
        del outputs, inputs
        torch.cuda.empty_cache()

    return text, question, full_prompt

def run_video_pipeline():
    metadata_path = os.path.join(DATA_ROOT, "metadata.jsonl")
    print("\n=== [VIDEO] Loading local metadata ===")
    print("Metadata path:", metadata_path)

    ds = load_dataset("json", data_files=metadata_path, split="train")
    print(f"Loaded {len(ds)} video rows.")

    results = []

    for idx, row in enumerate(ds):
        print(f"\n===== [VIDEO] Running sample {idx} / {len(ds)} =====")

        try:
            model_answer, question, full_prompt = run_one_video(row)
        except Exception as e:
            model_answer = f"ERROR: {e}"
            question = row.get("question", None)
            full_prompt = None

        video_rel = row.get("video_path", None)
        if video_rel is not None:
            video_full_path = os.path.join(DATA_ROOT, video_rel)
        else:
            video_full_path = None

        results.append({
            "id": row.get("id", None),
            "split": row.get("split", None),
            "video": video_full_path,
            "video_path": video_rel,
            "question": question,
            "answer": row.get("answer", None),
            "duration_sec": row.get("duration_sec", None),
            "source": row.get("source", None),
            "model_answer": model_answer,
        })

        print("Question:", question)
        print("Model answer:\n", model_answer)

    df = pd.DataFrame(results)
    parquet_path = "phi4_video_results.parquet"
    df.to_parquet(parquet_path)
    print(f"\n[VIDEO] Saved results to: {parquet_path}")

    hf_ds = Dataset.from_pandas(df, preserve_index=False)
    hf_ds = hf_ds.cast_column("video", Video())
    hf_ds.push_to_hub("yulianawu/phi4_video")
    print("[VIDEO] Upload complete! Dataset: https://huggingface.co/datasets/yulianawu/phi4_video")


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run Phi-4-Multimodal on audio/image/text(tt)/video datasets."
    )
    parser.add_argument(
        "--mode",
        choices=["audio", "image", "tt", "video", "all"],
        required=True,
        help="Which pipeline to run.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=os.getenv("DATA_ROOT", DATA_ROOT),
        help="Directory containing video files and metadata.jsonl.",
    )
    args = parser.parse_args()
    
    global DATA_ROOT
    DATA_ROOT = args.data_root

    if args.mode in ("audio", "all"):
        run_audio_pipeline()
    if args.mode in ("image", "all"):
        run_image_pipeline()
    if args.mode in ("tt", "all"):
        run_tt_pipeline()
    if args.mode in ("video", "all"):
        run_video_pipeline()


if __name__ == "__main__":
    main()
