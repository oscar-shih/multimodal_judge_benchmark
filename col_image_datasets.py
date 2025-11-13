# ====== Image QA Collector + Push to Hugging Face Hub ======
import os, io, re, json, shutil, hashlib, random, requests, gc
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
from PIL import Image
from datasets import (
    load_dataset, Dataset, DatasetDict,
    Features, Value, Sequence
)
from datasets import Image as HFImage
from huggingface_hub import HfApi, HfFolder

# ---------- Config ----------
SEED = 42
random.seed(SEED)
LETTER_TAGS = list("ABCDEFGH")
OUT_DIR = "out_all_imgqa"

REPO_ID = "JerrySprite/Image-Dataset-FP"

# ---------- Unified row ----------
@dataclass
class UniRow:
    uid: str
    source: str
    domain: str
    split: str
    question: Optional[str]
    choices: Optional[List[str]]
    answer: Optional[Any]
    task: Optional[str] = None
    rationale: Optional[str] = None
    images: Optional[List[Any]] = None
    meta: Optional[dict] = None

# ---------- Writer ----------
class DatasetWriter:
    def __init__(self, out_dir: str, img_format: str = "png", jpeg_quality: int = 90):
        self.out_dir = Path(out_dir)
        self.img_dir = self.out_dir / "images"
        self.meta_path = self.out_dir / "metadata.jsonl"
        self.img_format = img_format.lower()
        self.jpeg_quality = int(jpeg_quality)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.existing_ids = self._load_existing_ids()

    def _load_existing_ids(self) -> set:
        ids = set()
        if self.meta_path.exists():
            with open(self.meta_path, "r", encoding="utf-8") as f:
                for ln in f:
                    if not ln.strip():
                        continue
                    try:
                        j = json.loads(ln)
                        if "id" in j:
                            ids.add(j["id"])
                    except Exception:
                        pass
        return ids

    @staticmethod
    def _stable_id(uid: str) -> str:
        return hashlib.sha1(uid.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def _validate_row(row: Dict[str, Any]) -> None:
        for k in ["uid", "source", "split", "question"]:
            if k not in row:
                raise ValueError(f"Missing field: {k}")
        if row.get("choices") is not None and isinstance(row.get("answer"), int):
            if not (0 <= row["answer"] < len(row["choices"])):
                raise ValueError("answer (index) out of range")

    def _save_image(self, pil_img: Image.Image, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.img_format in {"jpg", "jpeg"}:
            pil_img.convert("RGB").save(path, quality=self.jpeg_quality)
        else:
            pil_img.save(path)

    def _to_pil(self, img_obj):
        if isinstance(img_obj, Image.Image):
            return img_obj

        if isinstance(img_obj, dict):
            p = img_obj.get("path")
            if isinstance(p, str) and os.path.exists(p):
                return Image.open(p)
            b = img_obj.get("bytes")
            if b:
                try:
                    return Image.open(io.BytesIO(b))
                except Exception:
                    return None

        if isinstance(img_obj, str):
            if os.path.exists(img_obj):
                return Image.open(img_obj)
            candidate_repos = [
                "MathLLMs/MathVision",
                "ChartFoundation/ECDBench",
                "nyu-visionx/CV-Bench",
                "nirajandhakal/realworldqa",
            ]
            for rid in candidate_repos:
                try:
                    url = f"https://huggingface.co/datasets/{rid}/resolve/main/{img_obj}"
                    r = requests.get(url, timeout=15)
                    if r.status_code == 200:
                        return Image.open(io.BytesIO(r.content))
                except Exception:
                    continue
            return None


        try:
            import numpy as np
            if isinstance(img_obj, np.ndarray):
                return Image.fromarray(img_obj)
        except Exception:
            pass

        return None

    def append_row(self, row: Dict[str, Any]) -> Optional[str]:
        self._validate_row(row)
        sample_id = self._stable_id(row["uid"])
        if sample_id in self.existing_ids:
            return None

        images = row.get("images") or []
        saved_rel = []
        for i, img in enumerate(images):
            suffix = "" if len(images) == 1 else f"_{i}"
            rel = f"images/{sample_id}{suffix}.{self.img_format}"
            abs_p = self.out_dir / rel
            pil = self._to_pil(img)
            if pil is None:
                print(f"save image failed ({sample_id}, idx={i}): unsupported type {type(img)}")
                continue
            try:
                self._save_image(pil, abs_p)
                saved_rel.append(rel)
            except Exception as e:
                print(f"save image failed ({sample_id}, idx={i}): {e}")

        rec = {
            "id": sample_id,
            "image_paths": saved_rel,
            "question": row.get("question", ""),
            "choices": row.get("choices"),
            "answer": row.get("answer"),
            "task": row.get("task"),
            "rationale": row.get("rationale"),
            "meta": row.get("meta"),
            "uid": row.get("uid"),
            "source": row.get("source"),
            "domain": str(row.get("domain") or ""),
            "split": row.get("split"),
        }
        with open(self.meta_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.existing_ids.add(sample_id)
        return sample_id

# ---------- Helpers ----------
def rng_sample_indices(n_items: int, k: int, seed: int = SEED) -> List[int]:
    rng = random.Random(seed)
    idx = list(range(n_items))
    rng.shuffle(idx)
    return idx[:min(k, n_items)]

def _reservoir_sample(stream_iter, k: int, seed: int = SEED):
    rnd, res, n = random.Random(seed), [], 0
    for item in stream_iter:
        n += 1
        if len(res) < k:
            res.append(item)
        else:
            j = rnd.randint(1, n)
            if j <= k:
                res[j - 1] = item
    return res

# ============================================================
#                      CV-Bench
# ============================================================
def unify_cvbench_row(example: Dict[str, Any],
                      split: str,
                      row_id: int) -> Dict[str, Any]:
    question = example.get("question") or example.get("instruction") or example.get("prompt")
    options = example.get("choices") or example.get("options")

    if isinstance(options, list):
        options = [f"{LETTER_TAGS[i]}. {opt}" for i, opt in enumerate(options)]

    raw_ans = example.get("answer") or example.get("label") or example.get("gt_answer")
    answer = raw_ans
    if isinstance(raw_ans, int) and 0 <= raw_ans < len(LETTER_TAGS):
        answer = LETTER_TAGS[raw_ans]
    elif isinstance(raw_ans, str) and raw_ans.strip().upper() in LETTER_TAGS:
        answer = raw_ans.strip().upper()

    imgs = []
    if example.get("image") is not None:
        imgs = [example["image"]]
    elif isinstance(example.get("images"), list):
        imgs = [im for im in example["images"] if im is not None]

    uid = f"nyu-visionx/CV-Bench:{split}:{example.get('id', row_id)}"
    meta_keep = ["category", "subtask", "task", "level", "difficulty", "topic", "subject", "id"]
    meta = {k: example.get(k) for k in meta_keep if k in example}
    meta["raw_answer"] = raw_ans

    return UniRow(
        uid=uid,
        source="nyu-visionx/CV-Bench",
        domain="cvbench",
        split="train",                 
        question=question,
        choices=options,
        answer=answer,
        task="Image Captioning",       
        rationale=example.get("rationale") or example.get("solution"),
        images=imgs,
        meta=meta
    ).__dict__

def collect_cvbench(writer: DatasetWriter, n: int = 10, seed: int = SEED):
    sp = "test"
    try:
        ds = load_dataset("nyu-visionx/CV-Bench", split=sp)
        if "image" in ds.column_names:
            ds = ds.cast_column("image", HFImage())
    except Exception as e:
        print(f"Skip CV-Bench/{sp}: {e}")
        return

    rows = [unify_cvbench_row(ex, sp, i) for i, ex in enumerate(ds)]
    df = pd.DataFrame(rows)
    df = df[df["images"].map(lambda x: isinstance(x, list) and len(x) > 0)]
    if len(df) == 0:
        print("CV-Bench/test empty")
        return

    for _, r in df.iloc[rng_sample_indices(len(df), n, seed)].iterrows():
        writer.append_row(r.to_dict())
    print(f"CV-Bench  -> {writer.out_dir}")

# ============================================================
#                      ECDBench (streaming)
# ============================================================
_ECD_IMG_KEYS = [
    "base_image", "image", "images", "img", "img_path", "image_path",
    "chart", "chart_path", "figure", "fig"
] + [f"image_{i}" for i in range(1, 9)]

def _ecd_pick_images(example: Dict[str, Any]) -> List[Any]:
    imgs = []
    for k in _ECD_IMG_KEYS:
        if k in example and example[k] is not None:
            v = example[k]
            if isinstance(v, list):
                imgs.extend([vv for vv in v if vv is not None])
            else:
                imgs.append(v)
    return imgs

def unify_ecd_row_streaming(example: Dict[str, Any],
                            subset_hint: Optional[str],
                            split: str,
                            row_id: int) -> Dict[str, Any]:
    question = example.get("question") or example.get("prompt")
    answer = example.get("answer") or example.get("label") or example.get("gt_answer")
    images = _ecd_pick_images(example)

    subset_name = subset_hint or example.get("split")
    uid = f"ChartFoundation/ECDBench:{subset_name or 'unknown'}:{split}:{example.get('image_id', row_id)}"

    meta_keep = ["image_id", "category", "subtask", "task", "difficulty", "topic", "subject", "split"]
    meta = {k: example.get(k) for k in meta_keep if k in example}
    dom = f"ecdb_{subset_name}" if subset_name else "ecdb"

    return UniRow(
        uid=uid,
        source="ChartFoundation/ECDBench",
        domain=dom,
        split="train",                   
        question=question,
        choices=None,
        answer=answer,
        task="Chart Understanding",
        rationale=example.get("rationale") or example.get("solution"),
        images=images,
        meta=meta
    ).__dict__

def collect_ecdbench(writer: DatasetWriter,
                     n: int = 15,
                     seed: int = SEED,
                     stream: bool = True):
    """ECDBench: default + descriptive + reasoning"""
    all_rows_stream = []

    def _yield_rows(dsid: str, name: Optional[str]):
        sp = "test"
        kwargs = {"split": sp, "streaming": stream}
        if name is not None:
            ds = load_dataset(dsid, name=name, **kwargs)
            subset_hint = name
        else:
            ds = load_dataset(dsid, **kwargs)
            subset_hint = None
        for i, ex in enumerate(ds):
            row = unify_ecd_row_streaming(ex, subset_hint, sp, i)
            if isinstance(row["images"], list) and len(row["images"]) > 0:
                yield row

    for name in [None, "descriptive", "reasoning"]:
        try:
            gen = _yield_rows("ChartFoundation/ECDBench", name)
            all_rows_stream.extend(_reservoir_sample(gen, max(n, 20), seed))
        except Exception:
            continue

    if not all_rows_stream:
        print("ECDBench: no rows collected")
        return

    pick_idx = rng_sample_indices(len(all_rows_stream), n, seed)
    for idx in pick_idx:
        writer.append_row(all_rows_stream[idx])

    print(f"ECDBench -> {writer.out_dir}")

# ============================================================
#                      MathVision
# ============================================================
_MV_IMG_KEYS = ["image", "images", "figure", "fig", "img", "img_path", "image_path"] + [
    f"image_{i}" for i in range(1, 9)
]
_MV_LETTERS = list("ABCDEFGH")

def _mv_pick_images(example: Dict[str, Any]) -> List[Any]:
    if example.get("decoded_image") is not None:
        return [example["decoded_image"]]
    imgs = []
    for k in _MV_IMG_KEYS:
        if k in example and example[k] is not None:
            v = example[k]
            if isinstance(v, list):
                imgs.extend([vv for vv in v if vv is not None])
            else:
                imgs.append(v)
    return imgs

def _mv_norm_choices_and_answer(example: Dict[str, Any]):
    raw_choices = example.get("choices") or example.get("options")
    raw_ans = (
        example.get("answer") or example.get("final_answer")
        or example.get("label") or example.get("gt_answer")
    )

    if isinstance(raw_choices, list) and len(raw_choices) > 0:
        choices = [f"{_MV_LETTERS[i]}. {c}" for i, c in enumerate(raw_choices)]
        if isinstance(raw_ans, int) and 0 <= raw_ans < len(_MV_LETTERS):
            ans = _MV_LETTERS[raw_ans]
        elif isinstance(raw_ans, str) and raw_ans.strip().upper() in _MV_LETTERS:
            ans = raw_ans.strip().upper()
        else:
            ans = raw_ans
        return choices, ans
    else:
        return None, raw_ans

def unify_mathvision_row(example: Dict[str, Any],
                         split: str,
                         row_id: int) -> Dict[str, Any]:
    question = example.get("question") or example.get("prompt") or example.get("problem")
    images = _mv_pick_images(example)
    choices, ans = _mv_norm_choices_and_answer(example)

    uid = f"MathLLMs/MathVision:{split}:{example.get('id', row_id)}"

    meta_keep = ["id", "difficulty", "level", "topic", "subtopic", "year", "grade", "subject"]
    meta = {k: example.get(k) for k in meta_keep if k in example}
    meta["raw_answer"] = (
        example.get("answer") or example.get("final_answer")
        or example.get("label") or example.get("gt_answer")
    )

    return UniRow(
        uid=uid,
        source="MathLLMs/MathVision",
        domain="mathvision",
        split="train",                  
        question=question,
        choices=choices,
        answer=ans,
        task="Math Reasoning",
        rationale=example.get("solution") or example.get("rationale"),
        images=images,
        meta=meta
    ).__dict__

def collect_mathvision(writer: DatasetWriter,
                       n: int = 15,
                       seed: int = SEED):
    sp = "test"
    try:
        ds = load_dataset("MathLLMs/MathVision", split=sp)
        if "decoded_image" in ds.column_names:
            try:
                ds = ds.cast_column("decoded_image", HFImage())
            except Exception as e:
                print(f"cast decoded_image failed for MathVision: {e}")
        elif "image" in ds.column_names:
            try:
                ds = ds.cast_column("image", HFImage())
            except Exception as e:
                print(f"cast image failed for MathVision: {e}")
    except Exception as e:
        print(f"Skip MathVision/{sp}: {e}")
        return

    rows = [unify_mathvision_row(ex, sp, i) for i, ex in enumerate(ds)]
    df = pd.DataFrame(rows)
    df = df[df["images"].map(lambda x: isinstance(x, list) and len(x) > 0)]
    if len(df) == 0:
        print("MathVision/test empty after image filter")
        return

    for _, r in df.iloc[rng_sample_indices(len(df), n, seed)].iterrows():
        writer.append_row(r.to_dict())

    print(f"MathVision  -> {writer.out_dir}")

# ============================================================
#                      RealWorldQA (streaming)
# ============================================================
_RWQ_IMG_KEYS = ["decoded_image", "image", "img", "image_path", "picture", "photo"] + [
    f"image_{i}" for i in range(1, 5)
]
_RWQ_LETTERS = list("ABCDEFGH")

def _rwq_pick_images(ex: dict):
    if ex.get("decoded_image") is not None:
        return [ex["decoded_image"]]
    imgs = []
    for k in _RWQ_IMG_KEYS:
        if k in ex and ex[k] is not None:
            v = ex[k]
            if isinstance(v, list):
                imgs.extend([vv for vv in v if vv is not None])
            else:
                imgs.append(v)
    return imgs

def _rwq_norm_choices_and_answer(ex: dict):
    raw_choices = ex.get("choices") or ex.get("options") or ex.get("candidates")
    raw_ans = ex.get("answer") or ex.get("label") or ex.get("gt_answer")
    if isinstance(raw_choices, list) and len(raw_choices) > 0:
        choices = [f"{_RWQ_LETTERS[i]}. {c}" for i, c in enumerate(raw_choices)]
        if isinstance(raw_ans, int) and 0 <= raw_ans < len(_RWQ_LETTERS):
            ans = _RWQ_LETTERS[raw_ans]
        elif isinstance(raw_ans, str) and raw_ans.strip().upper() in _RWQ_LETTERS:
            ans = raw_ans.strip().upper()
        else:
            ans = raw_ans
        return choices, ans
    else:
        return None, raw_ans

def unify_realworldqa_row(example: dict,
                          split: str,
                          row_id: int) -> Dict[str, Any]:
    question = example.get("question") or example.get("prompt") or example.get("query")
    images = _rwq_pick_images(example)
    choices, ans = _rwq_norm_choices_and_answer(example)
    uid = f"nirajandhakal/realworldqa:{split}:{example.get('id', row_id)}"
    keep = ["id", "category", "topic", "scene", "source", "difficulty"]
    meta = {k: example.get(k) for k in keep if k in example}
    return UniRow(
        uid=uid,
        source="nirajandhakal/realworldqa",
        domain="realworldqa",
        split="train",                  
        question=question,
        choices=choices,
        answer=ans,
        task="Image Captioning",
        rationale=example.get("rationale") or example.get("explanation"),
        images=images,
        meta=meta
    ).__dict__

def collect_realworldqa(writer: DatasetWriter,
                        n: int = 10,
                        seed: int = SEED):
    import random
    random.seed(seed)

    used_split = None
    try:
        ds = load_dataset("nirajandhakal/realworldqa", split="test", streaming=True)
        used_split = "test"
    except Exception:
        try:
            ds = load_dataset("nirajandhakal/realworldqa", split="validation", streaming=True)
            used_split = "validation"
        except Exception as e:
            print(f"Skip RealWorldQA: cannot stream test/validation: {e}")
            return

    k = n
    reservoir = []
    total = 0

    for i, ex in enumerate(ds):
        total += 1
        row = unify_realworldqa_row(ex, used_split, i)

        imgs = row.get("images") or []
        if not (isinstance(imgs, list) and len(imgs) > 0):
            continue

        if len(reservoir) < k:
            reservoir.append((row, i))
        else:
            j = random.randint(0, i)
            if j < k:
                reservoir[j] = (row, i)

        if i % 500 == 0:
            gc.collect()

    if not reservoir:
        print("RealWorldQA empty after streaming filter")
        return

    for row, _idx in reservoir:
        writer.append_row(row)

    print(f"RealWorldQA  (streaming) -> {writer.out_dir} | scanned={total}, kept={len(reservoir)}")

# ============================================================
#                      Push to Hub
# ============================================================
def push_to_hub(out_dir: str, repo_id: str, private: bool = False):
    meta_path = Path(out_dir) / "metadata.jsonl"
    if not meta_path.exists():
        print(f"[!] {meta_path} not found")
        return

    recs = []
    with meta_path.open("r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            rec = json.loads(ln)
            imgs = rec.get("image_paths") or []
            if not imgs:
                continue
            first_img = str(Path(out_dir) / imgs[0])
            choices = rec.get("choices")
            if isinstance(choices, list) and len(choices) == 0:
                choices = None
            recs.append({
                "id": rec.get("id"),
                "question": rec.get("question") or "",
                "choices": choices,
                "answer": str(rec.get("answer") or ""),
                "task": rec.get("task") or "",
                "image": first_img,
                "domain": rec.get("domain") or "",
                "source": rec.get("source") or "",
                "split": rec.get("split") or "",
            })

    print(f"Loaded {len(recs)} samples for upload.")

    if not recs:
        print("No records to upload.")
        return

    features = Features({
        "id": Value("string"),
        "question": Value("string"),
        "choices": Sequence(Value("string")),
        "answer": Value("string"),
        "task": Value("string"),
        "image": HFImage(),
        "domain": Value("string"),
        "source": Value("string"),
        "split": Value("string"),
    })

    ds = Dataset.from_list(recs, features=features)
    ds = ds.cast_column("image", HFImage())
    dd = DatasetDict({"train": ds})

    print("Pushing to Hugging Face Hub...")
    dd.push_to_hub(repo_id, private=private)
    print(f"Uploaded: https://huggingface.co/datasets/{repo_id}")

# ============================================================
#                      Main
# ============================================================
def main():
    shutil.rmtree(OUT_DIR, ignore_errors=True)
    writer = DatasetWriter(OUT_DIR)

    collect_cvbench(writer, n=10, seed=SEED)
    collect_ecdbench(writer, n=15, seed=SEED, stream=True)
    collect_mathvision(writer, n=15, seed=SEED)
    collect_realworldqa(writer, n=10, seed=SEED)

    print("Done collecting all datasets!")
    push_to_hub(OUT_DIR, REPO_ID, private=False)

if __name__ == "__main__":
    main()
