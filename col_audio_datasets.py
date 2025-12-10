from huggingface_hub import login
login()

from datasets import load_dataset, Features, Audio as HFAudio, Value, concatenate_datasets
from datasets import Dataset
from itertools import islice
import json
import random

audiocaps_ds = load_dataset("kuanhuggingface/audiocaps_hallucination", split="test")

audiocaps_ds = audiocaps_ds.shuffle(seed=42).select(range(10))

audiocaps_ds = audiocaps_ds.remove_columns([c for c in audiocaps_ds.column_names if c not in ["audio", "caption"]])

def cap_to_str(ex):
    c = ex.get("caption")
    if isinstance(c, list):
        ex["caption"] = c[0] if c else ""
    else:
        ex["caption"] = "" if c is None else str(c)
    return ex

audiocaps_ds = audiocaps_ds.map(cap_to_str)

audiocaps_ds = audiocaps_ds.cast(Features({
    "audio": HFAudio(decode=False),
    "caption": Value("string"),
}))

if "source" not in audiocaps_ds.column_names:
    audiocaps_ds = audiocaps_ds.map(lambda e: {"source": "audiocaps"})

audiocaps_ds = audiocaps_ds.cast(Features({
    "audio": HFAudio(decode=False),
    "caption": Value("string"),
    "source": Value("string"),
}))

bird_ds = load_dataset("tglcourse/5s_birdcall_samples_top20", split="train")

bird_ds = bird_ds.shuffle(seed=43).select(range(10))

bird_ds = bird_ds.cast(Features({
    "audio": HFAudio(decode=False),
    "label": Value("string"),
}))

bird_ds = bird_ds.map(lambda e: {"caption": e["label"], "source": "birdcall"})

bird_ds = bird_ds.remove_columns([c for c in bird_ds.column_names if c not in ["audio","caption","source"]])

bird_ds = bird_ds.cast(Features({
    "audio": HFAudio(decode=False),
    "caption": Value("string"),
    "source": Value("string"),
}))

suno_stream = load_dataset("nyuuzyou/suno", split="train", streaming=True)

audio_list, caption_list, source_list = [], [], []

for row in suno_stream:
    if random.random() < 0.05:
        url = row.get("audio_url") or ""
        audio_list.append({"path": url})

        meta = row.get("metadata")
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {"tags": meta}
        if not isinstance(meta, dict):
            meta = {}

        raw = meta.get("tags")
        if isinstance(raw, list):
            tags = ", ".join(map(str, raw))
        elif isinstance(raw, str):
            tags = raw
        else:
            tags = meta.get("prompt", "") or ""

        caption_list.append(tags)
        source_list.append("suno")

        if len(audio_list) >= 10:
            break

suno_ds = Dataset.from_dict(
    {
        "audio_path": [d["path"] for d in audio_list],
        "caption": caption_list,
        "source":  source_list,
    },
    features=Features({
        "audio_path": Value("string"),
        "caption": Value("string"),
        "source":  Value("string"),
    })
)
suno_ds = suno_ds.map(lambda e: {"audio": {"path": e["audio_path"]}})
suno_ds = suno_ds.remove_columns(["audio_path"])
features = Features({
    "audio": HFAudio(decode=False),
    "caption": Value("string"),
    "source":  Value("string"),
})

suno_ds = suno_ds.cast(features)

combined = concatenate_datasets([audiocaps_ds, bird_ds, suno_ds])

combined = combined.map(
    lambda e, idx: {"id": idx + 1},
    with_indices=True
)

new_column_order = ["id"] + [col for col in combined.column_names if col not in ["id"]]

combined = combined.select_columns(new_column_order)

if "question" not in combined.column_names:
    combined = combined.add_column("question", [""] * len(combined))

features = Features({
    "id": Value("int64"),
    "audio": HFAudio(decode=False),
    "question": Value("string"),
    "caption": Value("string"),
    "source": Value("string"),
})

combined = combined.cast(features)

def add_question_for_id_1_to_7(example, idx):
    if 0 <= idx < 7:
        example["question"] = (
            "Please describe what can be heard in this audio."
        )
    return example

combined = combined.map(add_question_for_id_1_to_7, with_indices=True)

def add_question_for_id_8_to_10(example, idx):
    if 7 <= idx < 10:
        example["question"] = (
            "Does this audio contain bird sounds? If not, please describe what can be heard in this audio."
        )
    return example

combined = combined.map(add_question_for_id_8_to_10, with_indices=True)

def add_caption_for_id_8(example, idx):
    if idx == 7:
        example["caption"] = "No bird sounds. A voice on loudspeakers is drowned out by tires squealing and engines repeatedly revving."
    return example

combined = combined.map(add_caption_for_id_8, with_indices=True)

def add_caption_for_id_9(example, idx):
    if idx == 8:
        example["caption"] = "No bird sounds. A toilet flushing."
    return example

combined = combined.map(add_caption_for_id_9, with_indices=True)

def add_caption_for_id_10(example, idx):
    if idx == 9:
        example["caption"] = "No bird sounds. A woman speaking."
    return example

combined = combined.map(add_caption_for_id_10, with_indices=True)

def add_question_for_id_11_to_15(example, idx):
    if 10 <= idx < 15:
        example["question"] = """Please identify the bird species in this audio clip.
Answer with the eBird six-letter species code:

eursta - European Starling
houspa - House Sparrow
sonspa - Song Sparrow
howwre - House Wren
norcar - Northern Cardinal
spotow - Spotted Towhee
barswa - Barn Swallow
swathr - Swainson’s Thrush
nobird - (no bird sound)
"""
    return example

combined = combined.map(add_question_for_id_11_to_15, with_indices=True)

def add_question_for_id_16_to_18(example, idx):
    if 15 <= idx < 18:
        example["question"] = """Does this audio contain bird sounds? If yes, please identify the bird species.
Answer with the eBird six-letter species code:

eursta - European Starling
houspa - House Sparrow
sonspa - Song Sparrow
howwre - House Wren
norcar - Northern Cardinal
spotow - Spotted Towhee
barswa - Barn Swallow
swathr - Swainson’s Thrush
nobird - (no bird sound)
"""
    return example

combined = combined.map(add_question_for_id_16_to_18, with_indices=True)

def add_caption_for_id_16(example, idx):
    if idx == 15:
        example["caption"] = "Yes. eursta"
    return example

combined = combined.map(add_caption_for_id_16, with_indices=True)

def add_caption_for_id_17(example, idx):
    if idx == 16:
        example["caption"] = "Yes. norcar"
    return example

combined = combined.map(add_caption_for_id_17, with_indices=True)

def add_caption_for_id_18(example, idx):
    if idx == 17:
        example["caption"] = "Yes. spotow"
    return example

combined = combined.map(add_caption_for_id_18, with_indices=True)

def add_question_for_id_19_to_20(example, idx):
    if 18 <= idx < 20:
        example["question"] = "Please describe the musical style or genre of this audio clip."
    return example

combined = combined.map(add_question_for_id_19_to_20, with_indices=True)

def add_caption_for_id_19_to_20(example, idx):
    if 18 <= idx < 20:
        example["caption"] = "This recording does not contain music; it only includes bird sounds."
    return example

combined = combined.map(add_caption_for_id_19_to_20, with_indices=True)

def add_question_for_id_21_to_24(example, idx):
    if 20 <= idx < 24:
        example["question"] = "Please describe the musical style or genre of this audio clip."
    return example

combined = combined.map(add_question_for_id_21_to_24, with_indices=True)

def add_question_for_id_25_to_28(example, idx):
    if 24 <= idx < 28:
        example["question"] = "Please identify the bird species in this audio clip."
    return example

combined = combined.map(add_question_for_id_25_to_28, with_indices=True)

def add_caption_for_id_25_to_28(example, idx):
    if 24 <= idx < 28:
        example["caption"] = "This recording does not bird sounds."
    return example

combined = combined.map(add_caption_for_id_25_to_28, with_indices=True)

def add_question_for_id_29(example, idx):
    if idx == 28:
        example["question"] = """Given the audio clip and the provided tag list, judge whether the audio is relevant to the tags.
        tag list: ["soft and slow", "female singer singing happy birthday"]"""
    return example

combined = combined.map(add_question_for_id_29, with_indices=True)

def add_caption_for_id_29(example, idx):
    if idx == 28:
        example["caption"] = "The audio fully matches all tags, representing female singer softly singing Happy Birthday."
    return example

combined = combined.map(add_caption_for_id_29, with_indices=True)

def add_question_for_id_30(example, idx):
    if idx == 29:
        example["question"] = """Given the audio clip and the provided tag list, judge whether the audio is relevant to the tags.
        tag list: ["female singer", "symphonic rock", "classical"]"""
    return example

combined = combined.map(add_question_for_id_30, with_indices=True)

def add_caption_for_id_30(example, idx):
    if idx == 29:
        example["caption"] = "The audio partially matches the tag. The audio is sung by a male singer, and the style is symphonic rock with classical elements."
    return example

combined = combined.map(add_caption_for_id_30, with_indices=True)

speech_ds = load_dataset("AudioLLMs/public_sg_speech_qa_test", split="test")

speech_ds = speech_ds.shuffle(seed=42).select(range(20))

speech_ds = speech_ds.rename_columns({
    "context": "audio",
    "instruction": "question",
    "answer": "caption"
})

speech_ds = speech_ds.add_column("source", ["speech_qa"] * len(speech_ds))

if "id" in speech_ds.column_names:
    speech_ds = speech_ds.remove_columns("id")

start_id = len(combined)

speech_ds = speech_ds.add_column("id", list(range(start_id + 1, start_id + 1 + len(speech_ds))))

target_features = Features({
    "id": Value("int64"),
    "audio": HFAudio(decode=False),
    "question": Value("string"),
    "caption": Value("string"),
    "source": Value("string"),
})

combined  = combined.cast(target_features)
speech_ds   = speech_ds.cast(target_features)

combined = concatenate_datasets([combined, speech_ds])

def assign_task(example):
    if example["id"] in list(range(11, 19)) + [29, 30]:
        example["task"] = "Instruction-Following"
    else:
        example["task"] = "Question-Answer"
    return example

combined = combined.map(assign_task)

new_column_order = ["id", "task"] + [col for col in combined.column_names if col not in ["id", "task"]]
combined = combined.select_columns(new_column_order)

TARGET_REPO = "multi-judge/Audio_datasets"

combined.push_to_hub(
    TARGET_REPO,
    split="train",
    commit_message="Add data from 4 datasets"
)