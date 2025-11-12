from datasets import load_dataset, Dataset, load_dataset_builder, DatasetDict
from huggingface_hub import HfApi
import random, os, json
from typing import List, Dict, Any, Optional

SEED = 20010405
rng = random.Random(SEED)

def try_license(dataset_id: str) -> str:
    try:
        info = load_dataset_builder(dataset_id).info
        if getattr(info, "license", None):
            return str(info.license)
    except Exception:
        pass
    return "unknown"

def pick(ds, k: int):
    if isinstance(ds, DatasetDict):
        raise TypeError("Expected a single split `Dataset`, got `DatasetDict`. Did you forget `split=...`?")
    k = min(k, len(ds))
    return ds.shuffle(seed=SEED).select(range(k))

def load_any_split(dsid: str, subset: Optional[str] = None, preferred=("test","validation","train","split_0")):
    last_err = None
    for sp in preferred:
        try:
            if subset:
                return load_dataset(dsid, subset, split=sp), sp
            else:
                return load_dataset(dsid, split=sp), sp
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Unable to load any split for {dsid} ({subset or 'default'}): {last_err}")

def to_row(
    *,
    id: str,
    task: str,
    question: str,
    answer: str = "",
    options: Optional[List[str]] = None,
    context: Optional[str] = None,
    reasoning: Optional[str] = None,
    answer_type: str,
    source: str,
    source_url: str,
    split: str,
    license_str: str,
    constraints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "id": id,
        "task": task,
        "question": question,
        "answer": answer or "",
        "source": source,
        "reasoning": reasoning,
        "options": options,
        "context": context,
        "answer_type": answer_type,
        "source_url": source_url,
        "split": split,
        "license": license_str,
        "constraints": constraints,
    }

ALL_ROWS: List[Dict[str, Any]] = []

def collect_opencode_reasoning(n=20):
    dsid = "nvidia/OpenCodeReasoning"
    url = f"https://huggingface.co/datasets/{dsid}"
    lic = try_license(dsid)
    ds = load_dataset(dsid, "split_0", split="split_0"); split_name = "split_0"
    for ex in pick(ds, n):
        ALL_ROWS.append(to_row(
            id=f"OCR/{ex.get('id', '') or ex.get('hash', '')}",
            task="Reasoning-Code",
            question=(ex.get("input") or "").strip(),
            answer=(ex.get("solution") or "").strip(),
            reasoning=ex.get("output"),
            options=None,
            context=None,
            answer_type="code",
            source=dsid,
            source_url=url,
            split=split_name,
            license_str=lic,
        ))

def collect_openmath_reasoning(n=20):
    dsid = "nvidia/OpenMathReasoning"
    url = f"https://huggingface.co/datasets/{dsid}"
    lic = try_license(dsid)
    try:
        ds = load_dataset(dsid, split="cot"); split_name = "cot"
    except Exception:
        ds, split_name = load_any_split(dsid, subset=None)
    for i, ex in enumerate(pick(ds, n)):
        q = (ex.get("problem") or ex.get("question") or "").strip()
        ans = (ex.get("expected_answer") or ex.get("final_answer") or "").strip()
        reas = ex.get("generated_solution") or ex.get("rationale")
        ALL_ROWS.append(to_row(
            id=f"OMR/{split_name}/{ex.get('id', i)}",
            task="Reasoning-Math",
            question=q,
            answer=ans,
            reasoning=reas,
            options=None,
            context=None,
            answer_type="generative",
            source=dsid,
            source_url=url,
            split=split_name,
            license_str=lic,
        ))

def collect_gpqa_diamond(n=10, prefer_official=True):
    try:
        dsid = "Idavidrein/gpqa"
        url = f"https://huggingface.co/datasets/{dsid}"
        lic = try_license(dsid)
        ds = load_dataset(dsid, "diamond", split="test"); split_name = "test"
    except Exception:
        dsid = "fingertap/GPQA-Diamond"
        url = f"https://huggingface.co/datasets/{dsid}"
        lic = try_license(dsid)
        ds, split_name = load_any_split(dsid, subset=None, preferred=("test","validation","train"))
    for i, ex in enumerate(pick(ds, n)):
        q = ex.get("question") or ex.get("Question") or ""
        options = ex.get("options", None)
        ans = str(ex.get("answer") or ex.get("Answer") or "")
        ALL_ROWS.append(to_row(
            id=f"GPQA/diamond/{i}",
            task="Expert-MCQ",
            question=q.strip(),
            answer=ans.strip(),
            options=options,
            context=None,
            reasoning=None,
            answer_type="mc",
            source=dsid,
            source_url=url,
            split=split_name,
            license_str=lic,
        ))

def collect_drop(n=10):
    dsid = "ucinlp/drop"
    url = f"https://huggingface.co/datasets/{dsid}"
    lic = try_license(dsid)
    ds = load_dataset(dsid, split="validation"); split_name = "validation"
    for i, ex in enumerate(pick(ds, n)):
        spans = (ex.get("answers_spans") or {}).get("spans", [])
        ans = spans[0] if spans else ""
        ALL_ROWS.append(to_row(
            id=f"DROP/{split_name}/{i}",
            task="Reading-Comprehension",
            question=ex.get("question","").strip(),
            answer=str(ans),
            options=None,
            context=ex.get("passage"),
            reasoning=None,
            answer_type="extractive",
            source=dsid,
            source_url=url,
            split=split_name,
            license_str=lic,
        ))

def collect_hellaswag(n=15):
    dsid = "Rowan/hellaswag"
    url = f"https://huggingface.co/datasets/{dsid}"
    lic = try_license(dsid)
    ds = load_dataset(dsid, split="validation"); split_name = "validation"
    for i, ex in enumerate(pick(ds, n)):
        ctx = " ".join([ex.get("ctx_a",""), ex.get("ctx_b","")]).strip()
        ALL_ROWS.append(to_row(
            id=f"HellaSwag/{split_name}/{ex.get('ind', i)}",
            task="Commonsense-Reasoning",
            question=ctx,
            answer=str(ex.get("label","")),
            options=ex.get("endings"),
            context=None,
            reasoning=None,
            answer_type="mc",
            source=dsid,
            source_url=url,
            split=split_name,
            license_str=lic,
        ))

def collect_socialiqa(n=15):
    dsid = "jet-ai/social_i_qa"
    url = f"https://huggingface.co/datasets/{dsid}"
    lic = try_license(dsid)
    # use validation split for evaluation; you can swap to "train" if you prefer
    ds = load_dataset(dsid, split="validation"); split_name = "validation"

    # Optional sanity check on expected columns
    expected_cols = {"context", "question", "answerA", "answerB", "answerC", "label"}
    missing = expected_cols - set(ds.column_names)
    if missing:
        raise ValueError(f"{dsid}:{split_name} missing columns: {missing}")

    for i, ex in enumerate(pick(ds, n)):
        options = [ex.get("answerA",""), ex.get("answerB",""), ex.get("answerC","")]
        q = f"{(ex.get('context','').strip())} || Q: {(ex.get('question','').strip())}"
        ALL_ROWS.append(to_row(
            id=f"SocialIQA/{split_name}/{i}",
            task="Commonsense-Reasoning",
            question=q,
            answer=str(ex.get("label","")),     # "1", "2", or "3"
            options=options,
            context=None,                       # we already inlined context into question
            reasoning=None,
            answer_type="mc",
            source=dsid,
            source_url=url,
            split=split_name,
            license_str=lic,
        ))

def collect_ifeval(n=15):
    dsid = "google/IFEval"
    url = f"https://huggingface.co/datasets/{dsid}"
    lic = try_license(dsid)
    ds = load_dataset(dsid, split="train"); split_name = "train"
    for i, ex in enumerate(pick(ds, n)):
        constraints = {
            "instruction_id_list": ex.get("instruction_id_list", []),
            "kwargs": ex.get("kwargs", []),
            "key": ex.get("key"),
        }
        ALL_ROWS.append(to_row(
            id=f"IFEval/{split_name}/{ex.get('key', i)}",
            task="Instruction-Following",
            question=(ex.get("prompt","") or "").strip(),
            answer="",
            options=None,
            context=None,
            reasoning=None,
            answer_type="generative",
            source=dsid,
            source_url=url,
            split=split_name,
            license_str=lic,
            constraints=constraints,
        ))

def collect_followbench(n=15):
    dsid = "YuxinJiang/FollowBench"
    url = f"https://huggingface.co/datasets/{dsid}"
    lic = try_license(dsid)
    try:
        ds = load_dataset(dsid, split="train"); split_name = "train"
    except Exception:
        ds = load_dataset(dsid, split="validation"); split_name = "validation"
    for i, ex in enumerate(pick(ds, n)):
        q = ex.get("question") or ex.get("instruction") or ex.get("prompt") or ex.get("content") or ""
        a = ex.get("answer") or ex.get("output") or ex.get("reference") or ""
        constraints = {
            "type": ex.get("type") or ex.get("category"),
            "level": ex.get("level"),
            "meta": {k:v for k,v in ex.items() if k not in {"question","instruction","prompt","content","answer","output","reference"}}
        }
        ALL_ROWS.append(to_row(
            id=f"FollowBench/{split_name}/{i}",
            task="Instruction-Following",
            question=str(q).strip(),
            answer=str(a).strip(),
            options=None,
            context=None,
            reasoning=None,
            answer_type="generative",
            source=dsid,
            source_url=url,
            split=split_name,
            license_str=lic,
            constraints=constraints,
        ))

def collect_longbench_v2(n=20):
    dsid = "THUDM/LongBench-v2"
    url = "https://huggingface.co/datasets/zai-org/LongBench-v2"
    lic = try_license(dsid)
    ds = load_dataset(dsid, split="train"); split_name = "train"

    needed = {"_id", "question", "choice_A", "choice_B", "choice_C", "choice_D", "answer", "context"}
    missing = needed - set(ds.column_names)
    if missing:
        raise ValueError(f"{dsid}:{split_name} missing columns: {missing}")

    for i, ex in enumerate(pick(ds, n)):
        options = [ex["choice_A"], ex["choice_B"], ex["choice_C"], ex["choice_D"]]
        ALL_ROWS.append(to_row(
            id=f"LongBenchV2/{split_name}/{ex.get('_id', i)}",
            task="Long-Context",
            question=str(ex["question"]).strip(),
            answer=str(ex["answer"]).strip(),
            options=options,
            context=ex.get("context"),
            reasoning=None,
            answer_type="mc",
            source=dsid,
            source_url=url,
            split=split_name,
            license_str=lic,
        ))

def push_to_hub(repo_id: str, private: bool = True, filename: str = None, token: str = None):
    if not filename or not os.path.exists(filename):
        raise FileNotFoundError(filename or "<missing filename>")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True, repo_type="dataset")
    api.upload_file(
        path_or_fileobj=filename,
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"Pushed {filename} to hf://datasets/{repo_id} (private={private})")

# if __name__ == "__main__":
#     out_path = "mini_benchmark_150.jsonl"
#     if not os.path.exists(out_path):
#         collect_opencode_reasoning(20)
#         collect_openmath_reasoning(20)
#         collect_gpqa_diamond(10)
#         collect_drop(20)
#         collect_hellaswag(15)
#         collect_socialiqa(15)
#         collect_ifeval(15)
#         collect_followbench(15)
#         collect_longbench_v2(20)
#         print(f"Total collected: {len(ALL_ROWS)}")

#         with open(out_path, "w", encoding="utf-8") as f:
#             for row in ALL_ROWS:
#                 f.write(json.dumps(row, ensure_ascii=False) + "\n")
#         print(f"Saved -> {out_path}")

#     with open("api.json", "r", encoding="utf-8") as f:
#         api_keys = json.load(f)
#     hf_token = api_keys[0]["hf_token"]
#     push_to_hub("Oscarshih/ee599-tt-datasets", filename=out_path, private=True, token=hf_token)