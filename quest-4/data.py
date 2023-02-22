import datasets
from datasets import DatasetDict, load_dataset


def get_data(dataset_name) -> tuple:
    dataset: DatasetDict = load_dataset(dataset_name, cache_dir="./data/cache/huggingface/datasets")
    
    train = dataset["train"]
    train = filter_unlabelled_data(train)
    
    dev = dataset["validation"]
    dev = filter_unlabelled_data(dev)
    
    test = dataset["test"]
    test = filter_unlabelled_data(test)
    
    return train, dev, test


def filter_unlabelled_data(dataset, label=-1) -> datasets.Dataset:
    return dataset.filter(lambda example: example["label"] != label)