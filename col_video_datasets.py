# ============================================================
# video_collect.py
#   - Sample 50 videos from IntelligenceLab/VideoHallu
#   - Download each video locally
#   - Create metadata.jsonl (id, question, answer, video_path, duration)
#   - Build a HuggingFace DatasetDict(train=...) and push to Hub
#
# Requirements:
#   pip install datasets huggingface_hub fsspec opencv-python pillow
#
# HuggingFace login:
#   - Local: huggingface-cli login
#   - Colab: from huggingface_hub import notebook_login; notebook_login()
#
# ============================================================

from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import HfFolder
import os, json, shutil, fsspec, cv2
from pathlib import Path
import random

# ---------- Config ----------
DATASET_DIR = "/content/video_dataset"   
NUM_SAMPLES = 50                         
REPO_ID = "JerrySprite/Video-Dataset-FP"
SEED = 42

# ---------- Prepare output folder ----------
if os.path.exists(DATASET_DIR):
    shutil.rmtree(DATASET_DIR)
os.makedirs(f"{DATASET_DIR}/videos", exist_ok=True)
META_PATH = Path(DATASET_DIR) / "metadata.jsonl"


# ---------- Utility Functions ----------

def _hfuri_from_rel(rel: str) -> str:
    """Convert a relative path from the dataset into a valid hf:// URI."""
    rel = rel.lstrip("./")
    return f"hf://datasets/IntelligenceLab/VideoHallu/{rel}"


def _copy_hfuri_to_local(hfuri: str, dst_abs: str):
    """Copy a video from hf:// URI to a local file."""
    os.makedirs(os.path.dirname(dst_abs), exist_ok=True)
    with fsspec.open(hfuri, "rb") as fin, open(dst_abs, "wb") as fout:
        shutil.copyfileobj(fin, fout)


def _duration_sec(path: str) -> float:
    """Estimate video duration in seconds using OpenCV."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return (frames / fps) if fps > 0 else 0.0


def _next_id() -> str:
    """Generate incremental ID based on the number of metadata entries."""
    if META_PATH.exists():
        with open(META_PATH, "r", encoding="utf-8") as f:
            n = sum(1 for ln in f if ln.strip())
    else:
        n = 0
    return f"{n+1:04d}"


def _append_meta(rec: dict):
    """Append one metadata record into metadata.jsonl."""
    with open(META_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------- Main Collector: sample from validation + test ----------
def collect_all(k: int):
    """
    Collect k samples from both validation and test splits.
    All samples are stored under a unified split 'train'.
    """
    ds_all = []

    # Load both splits if available
    for split in ["validation", "test"]:
        try:
            ds = load_dataset("IntelligenceLab/VideoHallu", split=split)
            ds_all.extend(list(ds))
            print(f"Loaded {len(ds)} samples from split='{split}'")
        except Exception as e:
            print(f"⚠️ Could not load split '{split}': {e}")

    if not ds_all:
        raise RuntimeError("No samples loaded from IntelligenceLab/VideoHallu.")

    # Shuffle & sample k items
    random.seed(SEED)
    random.shuffle(ds_all)
    ds_all = ds_all[:k]

    for ex in ds_all:
        # Get video path reference
        rel = ex.get("video") or ex.get("video_path")
        if not isinstance(rel, str):
            continue

        # Convert to hf:// URI if needed
        if rel.startswith(("hf://", "http://", "https://", "zip::")):
            hfuri = rel
        else:
            hfuri = _hfuri_from_rel(rel)

        # Extract QA pair
        q = ex.get("Question") or ex.get("question")
        a = ex.get("Answer") or ex.get("answer")
        if not isinstance(q, str) or not isinstance(a, str):
            continue

        # Generate unique ID
        sid = _next_id()
        dst_rel = f"videos/{sid}.mp4"
        dst_abs = f"{DATASET_DIR}/{dst_rel}"

        # Download video & compute duration
        _copy_hfuri_to_local(hfuri, dst_abs)
        dur = round(_duration_sec(dst_abs), 3)

        # Write metadata
        rec = {
            "id": sid,
            "split": "train",           # unified split
            "video_path": dst_rel,
            "question": q.strip(),
            "answer": a.strip(),
            "duration_sec": dur,
            "source": "IntelligenceLab/VideoHallu",
        }
        _append_meta(rec)

        print(f"✔ Added sample {sid}  duration={dur:.2f}s")


# ---------- Build HF DatasetDict and push ----------
def build_and_push_hf_dataset():
    """Read metadata.jsonl → build DatasetDict(train=...) → push to HF Hub."""
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = [json.loads(l) for l in f if l.strip()]

    train_ds = Dataset.from_list(meta)
    dd = DatasetDict({"train": train_ds})

    print("Uploading dataset to HuggingFace Hub...")
    dd.push_to_hub(REPO_ID, private=False)

    print(f"Upload complete → https://huggingface.co/datasets/{REPO_ID}")


# ---------- main ----------
def main():
    print("=== Collecting VideoHallu samples ===")
    collect_all(NUM_SAMPLES)
    print("\nFinished collecting →", DATASET_DIR)

    build_and_push_hf_dataset()


if __name__ == "__main__":
    main()
