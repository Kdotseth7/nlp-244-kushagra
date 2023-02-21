import datasets
from datasets import DatasetDict, load_dataset

def get_data() -> tuple:
    dataset: DatasetDict = load_dataset("snli")
    
    train = dataset["train"]
    train = filter_unlabelled_data(train)
    
    dev = dataset["validation"]
    dev = filter_unlabelled_data(dev)
    
    test = dataset["test"]
    test = filter_unlabelled_data(test)
    
    return train, dev, test


def filter_unlabelled_data(dataset, label=-1) -> datasets.Dataset:
    return dataset.filter(lambda example: example["label"] != label)