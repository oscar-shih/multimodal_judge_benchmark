import os, json, random
import datasets
from datasets import load_dataset, Dataset, load_dataset_builder, DatasetDict
import transformers
from huggingface_hub import HfApi
from PIL import Image

def load_tt_datasets(dataset_name: str):
    ds = load_dataset(dataset_name)
    ds = ds.remove_columns(["id", "source","source_url", "split", "license"])
    return ds

def truncate_audio(ds, truncate_duration):
    def truncate_example(example):
        # Audio objects in datasets have a 'duration' attribute directly
        if hasattr(example['audio'], 'duration') and example['audio'].duration > truncate_duration:
            example['audio'] = example['audio'].truncate(truncate_duration)
        return example
    
    return ds.map(truncate_example)

def load_audio_datasets(dataset_name: str, truncate_duration: float, truncated: bool):
    ds = load_dataset(dataset_name)
    ds = ds.remove_columns(["id", "source"])
    # TODO: Some audio clips are too long, we need to truncate them. Need to test the longest truncation length for different models.
    if truncated:
        ds = truncate_audio(ds, truncate_duration)
    return ds

def resize_image(ds, resized_size):
    def resize_example(example):
        # example['image'] is already a PIL Image object, no need for Image.open()
        example['image'] = example['image'].resize(resized_size)
        return example
    
    return ds.map(resize_example)

def load_image_datasets(dataset_name: str, resized: bool, resized_size: (int, int)):
    ds = load_dataset(dataset_name)
    ds = ds.remove_columns(["id", "source"])
    if resized:
        ds = resize_image(ds, resized_size)
    return ds

def load_video_datasets(dataset_name: str):
    ds = load_dataset(dataset_name)
    # TODO: Check the columns of the dataset, and remove the columns that are not needed
    ds = ds.remove_columns(["id", "split", "source", "duration_sec"])
    return ds

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