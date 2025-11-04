import os, json, random
import datasets
from datasets import load_dataset, Dataset, load_dataset_builder, DatasetDict
import transformers
from huggingface_hub import HfApi

def load_tt_datasets(dataset_name: str, seed: int):
    ds = load_dataset(dataset_name)
    ds = ds.remove_columns(["id", "source","source_url", "split", "license"])
    ds = ds.shuffle(seed=seed)
    return ds

if __name__ == "__main__":
    seed = 1126
    ds = load_tt_datasets("Oscarshih/ee599-tt-datasets", seed)
    print(len(ds['train']))
    print(ds.column_names)
    print(ds['train'][0])