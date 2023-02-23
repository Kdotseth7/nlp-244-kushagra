import datasets
from datasets import DatasetDict, load_dataset
from torch.utils.data import Dataset


class SNLIDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index: int):
        example = self.examples[index]
        task_prefix = "Translate English to French:"
        premise = example['premise']
        hypothesis = example['hypothesis']
        premise_input = f"{task_prefix} {premise}"
        hypothesis_input = f"{task_prefix} {hypothesis}"
        label = example['label']
        return premise_input, hypothesis_input, label


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
