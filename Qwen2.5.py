#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Image, Video, Text & Audio QA inference script for Qwen2.5-Omni-3B.

Image dataset (e.g. USCECE/Image_datasets):
- image        (HF Image object)
- id           (str or int)
- question     (str)
- choices      (list[str], optional)
- answer       (str, ground-truth)
- task         (str, e.g. "Image Captioning", "Chart Understanding", "Math Reasoning", ...)
- source       (str, optional)

Video (local files + metadata.jsonl):
Each line in metadata.jsonl:
- id:          str
- video:       str  (relative or absolute path, e.g. "videos/xxx.mp4")
  (fallback keys: "video_path", "video_relpath")
- question:    str
- answer:      str (ground-truth)
- task:        str
- source:      str, optional

Text dataset (e.g. USCECE/tt_datasets):
- id:          str
- task:        str
- question:    str
- answer:      str (ground-truth)
- source:      str
- reasoning:   str (not used as prompt)
- options:     list[str] (optional multiple-choice)
- context:     str
- answer_type: str
- source_url:  str
- split:       str
- license:     str
- constraints: dict, optional

Audio dataset (e.g. USCECE/Audio_datasets):
- id:          int/str
- audio:       datasets.Audio ({"array": np.ndarray, "sampling_rate": int})
- question:    str
- caption:     str (ground-truth answer)
- task:        str
- source:      str

Model output format (ALL modes, enforced via prompt):
    Justification: ...
    Final Answer: ...

We parse into:
    {"answer": "...", "reason": "..."}
and save to JSONL (one record per line).
"""
import io
import os
import cv2
import numpy as np
import json
import argparse
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from datasets import Audio
import soundfile as sf

from PIL import Image

from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
    BitsAndBytesConfig,
)

# ===========================
# System prompts (per modality)
# ===========================

IMAGE_SYSTEM_PROMPT = """
You are a careful and concise vision-language model. Keep detailed reasoning internal. Use
ONLY the information visible in the image and question provided (and choices if any).
Do not hallucinate objects, text, or facts that cannot be confirmed from the image.
""".strip()

VIDEO_SYSTEM_PROMPT = """
You are a careful and concise vision-language model. Keep detailed reasoning internal.
You are answering a Video QA question. Answer the question ONLY based on the visual
content and events shown in the video.
""".strip()

TEXT_SYSTEM_PROMPT = """
You are a careful, concise model. Keep detailed reasoning internal. Use ONLY the
information provided here (and optional context). If context conflicts with prior knowledge,
prefer the context.
""".strip()

AUDIO_SYSTEM_PROMPT = """
You are a careful, concise model for audio-language reasoning. Use ONLY the provided
audio clip and text input (e.g., question) as evidence. Ignore prior knowledge and external
information. When audio cues conflict with textual descriptions, prefer the audio evidence.
For each task, you must:
- Carefully interpret sounds, voices, and background cues.
- Integrate audio perception with textual understanding when applicable.
- Avoid assumptions beyond what can be heard or explicitly stated.
""".strip()


# ===========================
# Prompt builders
# ===========================

def build_image_prompt(example: Dict[str, Any]) -> str:
    question = example.get("question", "")
    choices = example.get("choices", None)
    task_raw = example.get("task", None)

    parts: List[str] = []

    if task_raw == "Image Captioning":
        parts.append(
            "You are answering an Image Captioning question. "
            "If answer choices are provided, select exactly one choice.\n"
            "Rules:\n"
            "- Use only information visible in the image.\n"
            "- Keep the caption concise, factual, and contextually plausible.\n"
            "- If multiple-choice options are provided, select exactly one option."
        )
    elif task_raw == "Chart Understanding":
        parts.append(
            "You are answering a Chart Understanding question. "
            "Your answer must rely solely on the information explicitly shown in the chart.\n"
            "Rules:\n"
            "- Use only information visible in the chart (axes, labels, legends, values, markers).\n"
            "- Report numerical values exactly as displayed.\n"
            "- Do not infer or assume trends that are not explicitly shown."
        )
    elif task_raw == "Math Reasoning":
        parts.append(
            "You are answering a Math Reasoning question based on the visual diagram "
            "or information presented in the image.\n"
            "Rules:\n"
            "- Use only the mathematical information visible in the diagram or the problem statement.\n"
            "- Do not assume values, shapes, or relationships that are not explicitly shown.\n"
            "- If multiple-choice options are provided, select exactly one option."
        )
    else:
        parts.append(
            "You are solving a visual question answering task based on a single image. "
            "Carefully look at the image and answer based only on what you see."
        )

    if task_raw:
        parts.append(f"Task type: {task_raw}")

    parts.append(f"Question: {question}")

    if choices:
        choices = [c for c in choices if isinstance(c, str) and c.strip()]
        if choices:
            parts.append("Options:")
            for c in choices:
                parts.append(f"- {c}")
            parts.append(
                'If options are provided, choose the single best option and express it ONLY '
                'in the "Final Answer" line (e.g., "A", "B", "C").'
            )

    parts.append(
        "At the end, you MUST output EXACTLY two lines:\n"
        "Justification: <one short sentence that identifies the key visual or chart/mathematical evidence>\n"
        "Final Answer: <final short answer or option label>\n"
        "Do NOT output anything before or after these two lines."
    )

    return "\n".join(parts)


def build_video_prompt(example: Dict[str, Any]) -> str:
    question = example.get("question", "")
    task_raw = example.get("task", None)

    parts: List[str] = []

    parts.append(
        "Rules:\n"
        "- Use only the information visible in the video frames.\n"
        "- Consider the temporal order, motion, and interactions occurring across the clip.\n"
        "- Do not assume details that are not clearly shown."
    )

    if task_raw:
        parts.append(f"Task type: {task_raw}")

    parts.append(f"Question: {question}")

    parts.append(
        "At the end, you MUST output EXACTLY two lines:\n"
        "Justification: <Provide one sentence describing the key visual or temporal evidence that supports the answer>\n"
        "Final Answer: <short answer based only on the video>\n"
        "No extra text before/after those two lines."
    )

    return "\n".join(parts)


def build_text_prompt(example: Dict[str, Any]) -> str:
    question     = example.get("question", "")
    context      = example.get("context", "")
    options      = example.get("options", None)
    task_raw     = example.get("task", None)
    answer_type  = example.get("answer_type", None)
    constraints  = example.get("constraints", None)

    parts: List[str] = []

    # Task-specific instructions
    if task_raw == "Reasoning-Code":
        parts.append(
            "Reason about code if needed, but return ONLY the required final value/string "
            "(unless the prompt explicitly asks for code).\n"
            "Rules:\n"
            "- If a numeric/string result is required, return only that value.\n"
            "- Do not include code or stack traces.\n"
            "Justification: <brief description of the operation/algorithm that yields the value>\n"
            "Final Answer: <final_answer>"
        )
    elif task_raw == "Reasoning-Math":
        parts.append(
            "Solve the problem and output only the final canonical answer.\n"
            "Canonicalization:\n"
            "- Integers: plain integer (e.g., 42).\n"
            "- Fractions: lowest terms with \"/\" (e.g., 7/12).\n"
            "- Decimals: remove trailing zeros (e.g., 3.5 not 3.5000).\n"
            "- Sets/tuples: a,b,c or (1,2,3); sort elements.\n"
            "- Units: include if explicitly required."
        )
    elif task_raw == "Expert MCQ":
        parts.append(
            "You are answering a multiple-choice question. If a context is provided, treat it as ground truth. "
            "Choose exactly one option.\n"
            "Rules:\n"
            "- For justifications, cite a key phrase from the context OR briefly state the elimination logic.\n"
            "- Then output only the letter A, B, C, or D as the final answer.\n"
            "Justification: <why this option is supported or others are ruled out>\n"
            "Final Answer: <A|B|C|D>"
        )
    elif task_raw == "Reading Comprehension":
        parts.append(
            "Return the SHORTEST substring of the passage that exactly answers the question.\n"
            "The answer MUST be a verbatim span from the passage (case and punctuation preserved).\n"
            "Rules:\n"
            "- The answer must be a contiguous substring of the passage.\n"
            "- If multiple spans could work, return the shortest fully correct one.\n"
            "- For justifications, state why you find this span to answer this question.\n"
            "Justification: <quote a nearby clue or sentence index supporting the span>\n"
            "Final Answer: <exact span from passage>"
        )
    elif task_raw == "Commonsense-Reasoning":
        parts.append(
            "You are answering a multiple-choice question. If a context is provided, treat it as ground truth. "
            "Choose exactly one option.\n"
            "Rules:\n"
            "- For justifications, cite a key phrase from the context OR briefly state the elimination logic.\n"
            "- Then output only the letter A, B, C, or D as the final answer.\n"
            "Justification: <why this option is supported or others are ruled out>\n"
            "Final Answer: <A|B|C|D>"
        )
    elif task_raw and task_raw.startswith("Instruction Following"):
        parts.append(
            "Follow ALL constraints exactly. If constraints conflict, obey the most specific.\n"
            "Rules:\n"
            "- Respect formatting limits (e.g., length, JSON schema, allowed chars).\n"
            "- If a safety-violating constraint appears, return a safe alternative while still satisfying\n"
            "  neutral formatting constraints.\n"
            "Justification:\n"
            "- [ ] constraint1: pass/fail + 3–6 words\n"
            "- [ ] constraint2: pass/fail + 3–6 words\n"
            "... (tick [x] for pass, [ ] for fail)\n"
            "Final Answer: <final_answer>"
        )
    else:
        parts.append(
            "You are solving a text-only question answering or generation task. "
            "Use ONLY the information in the provided context and question (and options if given)."
        )

    if task_raw:
        parts.append(f"Task family: {task_raw}")
    if answer_type:
        parts.append(f"Expected answer type: {answer_type}")

    if isinstance(constraints, dict) and constraints:
        parts.append("Constraints (you MUST follow all of them):")
        for k, v in constraints.items():
            parts.append(f"- {k}: {v}")

    if context:
        parts.append("Context:")
        parts.append(context)

    parts.append(f"Question: {question}")

    if options:
        clean_opts = [o for o in options if isinstance(o, str) and o.strip()]
        if clean_opts:
            parts.append("Options:")
            for o in clean_opts:
                parts.append(f"- {o}")
            parts.append(
                "If options are provided, select exactly ONE best option and output only "
                "that option (or its label if specified) in the 'Final Answer' line."
            )

    parts.append(
        "At the end, output EXACTLY two lines:\n"
        "Justification: <Some sentences or a short checklist, task-specific>\n"
        "Final Answer: <...>\n"
        "No extra text before/after those two lines."
    )

    return "\n".join(parts)


def build_audio_prompt(example: Dict[str, Any]) -> str:
    question = example.get("question", "")
    task_raw = example.get("task", None)

    parts: List[str] = []

    parts.append(
        "You are solving an audio question answering task.\n"
        "Use ONLY the information from the audio clip and the question text.\n"
        "Do not use outside knowledge unless the question explicitly asks for it."
    )

    if task_raw == "Question-Answer":
        parts.append(
            "Please answer the following question based on the provided audio clip.\n"
            "Use ONLY the sounds in the audio and the accompanying text as evidence.\n"
            "Rules:\n"
            "- Carefully listen to the audio to identify relevant sounds, voices, or background events.\n"
            "- Base your response solely on what can be heard or clearly implied from the audio.\n"
            "- If no relevant sound is present, describe what is actually audible instead.\n"
            "Justification: <Short explanation based on audible cues or reasoning>\n"
            "Final Answer: <Concise answer or caption>"
        )
    elif task_raw == "Instruction-Following":
        parts.append(
            "Given the audio clip and the provided instruction, follow the instruction carefully and respond accordingly.\n"
            "Use ONLY the information from the audio and the given text input as evidence.\n"
            "Rules:\n"
            "- Listen carefully to the audio and identify cues related to the instruction.\n"
            "- Follow the instruction precisely.\n"
            "- Base your output only on what can be heard in the audio clip.\n"
            "Justification: <Brief reason or evidence from the audio supporting the output>\n"
            "Final Answer: <Concise completion or response>"
        )

    parts.append(f"Question: {question}")

    parts.append(
        "At the end, output EXACTLY two lines:\n"
        "Justification: <Short explanation based on audible cues or evidence>\n"
        "Final Answer: <...>\n"
        "No extra text before or after these two lines."
    )

    return "\n".join(parts)


# ===========================
# Helper: parse model output
# ===========================

def parse_model_output(text: str) -> Dict[str, Any]:
    """
    Parse model output of the form:

      Justification: ...
      Final Answer: ...

    into {"answer": "...", "reason": "..."}.
    """
    text = text.strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    justification = ""
    final_answer = ""

    for ln in lines:
        low = ln.lower()
        if low.startswith("justification:"):
            justification = ln.split(":", 1)[1].strip()
        elif low.startswith("final answer:"):
            final_answer = ln.split(":", 1)[1].strip()

    if final_answer:
        return {
            "answer": final_answer,
            "reason": justification,
        }

    # Fallback
    return {
        "answer": text,
        "reason": "Failed to parse 'Final Answer'; using raw output as answer.",
    }


# ===========================
# Helper: (optional) video frames loader
# ===========================

def load_video_frames(
    video_path: str,
    num_frames: int = 8,
    target_size: int = 224,
) -> List[Image.Image]:
    """
    (Unused in current pipeline, but kept for potential future use.)
    Load a video from disk and sample `num_frames` frames uniformly.
    Each frame is resized to (target_size, target_size) and returned as PIL Images (RGB).
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = num_frames

    indices = np.linspace(0, max(total_frames - 1, 0), num=num_frames, dtype=int)
    index_set = set(indices.tolist())

    frames: List[Image.Image] = []
    current_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if current_idx in index_set:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb).convert("RGB")
            img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
            frames.append(img)

        current_idx += 1
        if len(frames) >= num_frames:
            break

    cap.release()

    if not frames:
        img = Image.new("RGB", (target_size, target_size), (0, 0, 0))
        frames.append(img)

    return frames


# ===========================
# Image inference
# ===========================

def run_image_inference(
    model_name: str,
    dataset_name: str,
    dataset_split: str,
    output_path: str,
    max_samples: Optional[int] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
) -> None:

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[IMAGE] Using device: {device}")
    print(f"[IMAGE] Loading model: {model_name}")

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(model_name)
    model.disable_talker()
    model.eval()

    print(f"[IMAGE] Loading dataset: {dataset_name} (split={dataset_split})")
    ds = load_dataset(dataset_name, split=dataset_split)

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    print(f"[IMAGE] Total examples: {len(ds)}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    results = []

    for idx, example in enumerate(ds):
        example_id = example.get("id", idx)
        question = example.get("question", "")
        choices = example.get("choices", None)
        answer_gt = example.get("answer", None)
        task = example.get("task", None)
        img_obj = example.get("image", None)

        print(f"[IMAGE] [{idx+1}/{len(ds)}] id={example_id} task={task}")

        if img_obj is None:
            raise ValueError(f"Example id={example_id} has no 'image' field.")

        if isinstance(img_obj, Image.Image):
            img_obj = img_obj.convert("RGB")
            target_size = 224
            img_obj = img_obj.resize(
                (target_size, target_size),
                Image.Resampling.LANCZOS,
            )

        user_prompt = build_image_prompt(example)

        conversations = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": IMAGE_SYSTEM_PROMPT},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_obj},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        inputs = processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        ).to(device)

        input_ids = inputs["input_ids"]

        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0,
                "use_cache": False,
                "return_audio": False,
            }
            if temperature > 0:
                gen_kwargs["temperature"] = temperature

            gen_out = model.generate(**inputs, **gen_kwargs)

        if isinstance(gen_out, torch.Tensor):
            sequences = gen_out
        elif isinstance(gen_out, (list, tuple)):
            sequences = gen_out[0]
        elif hasattr(gen_out, "sequences"):
            sequences = gen_out.sequences
        else:
            raise TypeError(f"Unexpected generate() output type: {type(gen_out)}")

        generated_ids = sequences[:, input_ids.shape[1]:]
        generated_text = processor.batch_decode(
            generated_ids.cpu(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        parsed = parse_model_output(generated_text)

        record = {
            "id": example_id,
            "question": question,
            "choices": choices,
            "task": task,
            "source": example.get("source", None),
            "ground_truth": answer_gt,
            "prediction": parsed["answer"],
            "reason": parsed.get("reason", ""),
            "raw_output": generated_text,
        }
        results.append(record)

        del inputs, gen_out, sequences, generated_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[IMAGE] Saved predictions to: {output_path}")


# ===========================
# Text inference (4-bit if possible)
# ===========================

def run_text_inference(
    model_name: str,
    dataset_name: str,
    dataset_split: str,
    output_path: str,
    max_samples: Optional[int] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
) -> None:

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[TEXT] Using device: {device}")
    print(f"[TEXT] Loading model: {model_name}")

    # Try 4-bit; if bitsandbytes is not installed, fall back to bf16
    bnb_config = None
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        print("[TEXT] Using 4-bit quantization with BitsAndBytes.")
    except Exception as e:
        print("[TEXT] Warning: BitsAndBytesConfig / bitsandbytes not available, "
              "falling back to bf16 full precision:", e)
        bnb_config = None

    if bnb_config is not None:
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=bnb_config,
        )
    else:
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    processor = Qwen2_5OmniProcessor.from_pretrained(model_name)
    model.disable_talker()
    model.eval()

    print(f"[TEXT] Loading dataset: {dataset_name} (split={dataset_split})")
    ds = load_dataset(dataset_name, split=dataset_split)

    # 先截 max_samples（不再做任何 task 過濾）
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    print(f"[TEXT] Total examples: {len(ds)}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    results = []

    for idx, example in enumerate(ds):
        example_id   = example.get("id", idx)
        question     = example.get("question", "")
        context      = example.get("context", "")
        options      = example.get("options", None)
        answer_gt    = example.get("answer", None)
        task         = example.get("task", None)
        answer_type  = example.get("answer_type", None)
        reasoning_gt = example.get("reasoning", None)
        constraints  = example.get("constraints", None)

        print(f"[TEXT] [{idx+1}/{len(ds)}] id={example_id} task={task} answer_type={answer_type}")

        user_prompt = build_text_prompt(example)

        conversations = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": TEXT_SYSTEM_PROMPT},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        inputs = processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        ).to(device)

        input_ids = inputs["input_ids"]

        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0,
                "use_cache": True,
                "return_audio": False,
            }
            if temperature > 0:
                gen_kwargs["temperature"] = temperature

            gen_out = model.generate(**inputs, **gen_kwargs)

        if isinstance(gen_out, torch.Tensor):
            sequences = gen_out
        elif isinstance(gen_out, (list, tuple)):
            sequences = gen_out[0]
        elif hasattr(gen_out, "sequences"):
            sequences = gen_out.sequences
        else:
            raise TypeError(f"Unexpected generate() output type: {type(gen_out)}")

        generated_ids = sequences[:, input_ids.shape[1]:]
        generated_text = processor.batch_decode(
            generated_ids.cpu(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        parsed = parse_model_output(generated_text)

        record = {
            "id": example_id,
            "question": question,
            "context": context,
            "options": options,
            "task": task,
            "answer_type": answer_type,
            "constraints": constraints,
            "source": example.get("source", None),
            "source_url": example.get("source_url", None),
            "ground_truth": answer_gt,
            "ground_truth_reasoning": reasoning_gt,
            "prediction": parsed["answer"],
            "reason": parsed.get("reason", ""),
            "raw_output": generated_text,
        }
        results.append(record)

        del inputs, gen_out, sequences, generated_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[TEXT] Saved predictions to: {output_path}")




# ===========================
# Video inference (metadata + local videos)
# ===========================

def run_video_inference_from_metadata(
    model_name: str,
    metadata_path: str,
    video_root: str,
    output_path: str,
    max_samples: Optional[int] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    num_frames: int = 8,
) -> None:

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[VIDEO] Using device: {device}")
    print(f"[VIDEO] Loading model: {model_name}")

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(model_name)
    model.disable_talker()
    model.eval()

    # load metadata.jsonl
    examples = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))

    if max_samples is not None:
        examples = examples[:max_samples]

    print(f"[VIDEO] Total video examples: {len(examples)}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    results = []

    for idx, ex in enumerate(examples):
        example_id = ex.get("id", idx)
        question   = ex.get("question", "")
        answer_gt  = ex.get("answer", None)
        task       = ex.get("task", None)

        video_rel = (
            ex.get("video_path")
            or ex.get("video")
            or ex.get("video_relpath")
        )
        if video_rel is None:
            raise ValueError(f"Example id={example_id} has no video path field.")

        if os.path.isabs(video_rel):
            video_path = video_rel
        else:
            video_path = os.path.join(video_root, video_rel)

        print(f"[VIDEO] [{idx+1}/{len(examples)}] id={example_id} task={task} video={video_path}")

        user_prompt = build_video_prompt(ex)

        conversations = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": VIDEO_SYSTEM_PROMPT},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        inputs = processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            load_audio_from_video=False,
            use_audio_in_video=False,
            num_frames=num_frames,
            padding=True,
        ).to(device)

        input_ids = inputs["input_ids"]

        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0,
                "use_cache": False,
                "return_audio": False,
            }
            if temperature > 0:
                gen_kwargs["temperature"] = temperature

            gen_out = model.generate(**inputs, **gen_kwargs)

        if isinstance(gen_out, torch.Tensor):
            sequences = gen_out
        elif isinstance(gen_out, (list, tuple)):
            sequences = gen_out[0]
        elif hasattr(gen_out, "sequences"):
            sequences = gen_out.sequences
        else:
            raise TypeError(f"Unexpected generate() output type: {type(gen_out)}")

        generated_ids = sequences[:, input_ids.shape[1]:]
        generated_text = processor.batch_decode(
            generated_ids.cpu(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        parsed = parse_model_output(generated_text)

        record = {
            "id": example_id,
            "question": question,
            "task": task,
            "source": ex.get("source", None),
            "ground_truth": answer_gt,
            "prediction": parsed["answer"],
            "reason": parsed.get("reason", ""),
            "raw_output": generated_text,
            "video_path": video_path,
        }
        results.append(record)

        del inputs, gen_out, sequences, generated_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[VIDEO] Saved predictions to: {output_path}")


# ===========================
# Audio inference (USCECE/Audio_datasets)
# ===========================
def load_hf_audio_bytes(audio_feature):
    """
    Convert HF Audio(decode=False) feature into (waveform, sampling_rate).
    Compatible with parquet datasets without torchcodec.
    """
    # audio_feature is a dict like:
    # {"bytes": b"...", "path": None}
    if not isinstance(audio_feature, dict) or "bytes" not in audio_feature:
        raise TypeError(f"Unexpected audio format: {type(audio_feature)} keys={audio_feature.keys()}")

    audio_bytes = audio_feature["bytes"]
    audio_file_obj = io.BytesIO(audio_bytes)

    # Use soundfile to decode
    audio_array, sampling_rate = sf.read(audio_file_obj, dtype="float32")

    # If stereo, convert to mono
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)

    return audio_array.astype("float32"), sampling_rate

def run_audio_inference(
    model_name: str,
    dataset_name: str,
    dataset_split: str,
    output_path: str,
    max_samples: Optional[int] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
) -> None:
    """
    Run Qwen2.5-Omni on an *audio* HF dataset.
    Dataset: USCECE/Audio_datasets
    Fields:
      - id:        int/str
      - audio:     datasets.Audio  (we will cast to Audio(decode=False))
      - question:  str
      - caption:   str (ground-truth answer)
      - task:      str
      - source:    str
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[AUDIO] Using device: {device}")
    print(f"[AUDIO] Loading model: {model_name}")

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(model_name)
    model.disable_talker()
    model.eval()

    print(f"[AUDIO] Loading dataset: {dataset_name} (split={dataset_split})")
    ds = load_dataset(dataset_name, split=dataset_split)

    # 如果要只跑前 n 個 sample，先 select 再 cast 也可以、反過來也行
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    # ⭐ 關鍵：關掉 HF 自動解碼，讓 audio 變成 bytes（避免 torchcodec）
    ds = ds.cast_column("audio", Audio(decode=False))

    print(f"[AUDIO] Total examples: {len(ds)}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    results = []

    for idx, example in enumerate(ds):
        example_id = example.get("id", idx)
        question   = example.get("question", "")
        task       = example.get("task", None)
        source     = example.get("source", None)
        answer_gt  = example.get("caption", None)  # ground-truth answer
        audio_feat = example.get("audio", None)    # {'bytes': ..., 'path': ...}

        print(f"[AUDIO] [{idx+1}/{len(ds)}] id={example_id} task={task} source={source}")

        if audio_feat is None:
            raise ValueError(f"Example id={example_id} has no 'audio' field.")

        # ========================
        # 1) Audio(decode=False) → waveform
        # ========================
        if not isinstance(audio_feat, dict) or "bytes" not in audio_feat:
            raise TypeError(f"Unexpected audio format for id={example_id}: {audio_feat}")

        audio_bytes = audio_feat["bytes"]
        audio_file_obj = io.BytesIO(audio_bytes)

        # 用 soundfile 解 bytes
        audio_array, sampling_rate = sf.read(audio_file_obj, dtype="float32")

        # 立體聲轉 mono
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)

        audio_array = audio_array.astype("float32")

        # ========================
        # 2) Build prompt
        # ========================
        user_prompt = build_audio_prompt(example)

        conversations = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": AUDIO_SYSTEM_PROMPT},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_array},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        # 交給 Qwen 的 processor，不需要自己處理 STFT 等
        inputs = processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        ).to(device)

        input_ids = inputs["input_ids"]

        # ========================
        # 3) Generate
        # ========================
        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0,
                "use_cache": True,
                "return_audio": False,
            }
            if temperature > 0:
                gen_kwargs["temperature"] = temperature

            gen_out = model.generate(**inputs, **gen_kwargs)

        # decode sequences
        if isinstance(gen_out, torch.Tensor):
            sequences = gen_out
        elif isinstance(gen_out, (list, tuple)):
            sequences = gen_out[0]
        elif hasattr(gen_out, "sequences"):
            sequences = gen_out.sequences
        else:
            raise TypeError(f"Unexpected generate() output type: {type(gen_out)}")

        generated_ids = sequences[:, input_ids.shape[1]:]
        generated_text = processor.batch_decode(
            generated_ids.cpu(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        parsed = parse_model_output(generated_text)

        record = {
            "id": example_id,
            "question": question,
            "task": task,
            "source": source,
            "ground_truth": answer_gt,
            "prediction": parsed["answer"],
            "reason": parsed.get("reason", ""),
            "raw_output": generated_text,
        }
        results.append(record)

        # 釋放一點記憶體
        del inputs, gen_out, sequences, generated_ids, audio_array
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[AUDIO] Saved predictions to: {output_path}")







# ========== CLI entry ==========

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Qwen2.5-Omni inference on image, video, text or audio QA datasets."
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["image", "video", "text", "audio"],
        default="image",
        help="Whether to run image QA (HF dataset), video QA (local metadata+files), text QA (HF dataset), or audio QA (HF dataset).",
    )

    # video-specific args
    parser.add_argument(
        "--video_metadata",
        type=str,
        default="metadata.jsonl",
        help="(VIDEO mode) Path to metadata.jsonl.",
    )
    parser.add_argument(
        "--video_root",
        type=str,
        default="videos",
        help="(VIDEO mode) Root directory that contains video files.",
    )
    parser.add_argument(
        "--video_num_frames",
        type=int,
        default=8,
        help="(VIDEO mode) Number of frames to sample from each video.",
    )

    # shared model / decoding args
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-Omni-3B",
        help="Hugging Face model name.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to the JSONL file where predictions will be saved.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="If set, run inference on at most this many samples.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature; 0 for greedy decoding.",
    )

    # dataset args (IMAGE / TEXT / AUDIO)
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="(IMAGE/TEXT/AUDIO mode) Hugging Face dataset name or local path.",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="(IMAGE/TEXT/AUDIO mode) Dataset split to use (e.g., train, validation, test).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "image":
        if args.dataset_name is None:
            raise ValueError("In image mode, --dataset_name must be provided.")
        run_image_inference(
            model_name=args.model_name,
            dataset_name=args.dataset_name,
            dataset_split=args.dataset_split,
            output_path=args.output_path,
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

    elif args.mode == "text":
        if args.dataset_name is None:
            raise ValueError("In text mode, --dataset_name must be provided.")
        run_text_inference(
            model_name=args.model_name,
            dataset_name=args.dataset_name,
            dataset_split=args.dataset_split,
            output_path=args.output_path,
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

    elif args.mode == "audio":
        if args.dataset_name is None:
            raise ValueError("In audio mode, --dataset_name must be provided.")
        run_audio_inference(
            model_name=args.model_name,
            dataset_name=args.dataset_name,
            dataset_split=args.dataset_split,
            output_path=args.output_path,
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

    else:  # video
        run_video_inference_from_metadata(
            model_name=args.model_name,
            metadata_path=args.video_metadata,
            video_root=args.video_root,
            output_path=args.output_path,
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            num_frames=args.video_num_frames,
        )
