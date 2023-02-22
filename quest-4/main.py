import torch
import os
import datasets
from torch.utils.data import Dataset, DataLoader
from utils import get_device
from data import get_data
from tqdm import tqdm
from transformers import (T5Config, 
                          T5Tokenizer, 
                          T5ForConditionalGeneration, 
                          TrainingArguments, 
                          Trainer)
from translate import SNLIDataset


def translate(loader: DataLoader, tokenizer: T5Tokenizer, model: T5ForConditionalGeneration, dataset_type: str) -> list:
    data_dict = dict()
    premise_translations = list()
    hypothesis_translations = list()
    labels = list()
    pbar = tqdm(loader, desc = f"Translating {dataset_type} Dataset using T5...", colour = 'red')
    for premise, hypothesis, label in pbar:
        premise_output_batch = model.generate(premise)
        premise_translations.extend(tokenizer.batch_decode(premise_output_batch, skip_special_tokens=True))
        hypothesis_output_batch = model.generate(hypothesis)
        hypothesis_translations.extend(tokenizer.batch_decode(hypothesis_output_batch, skip_special_tokens=True))
        labels.extend(label.tolist())
    data_dict["premise"] = premise_translations
    data_dict["hypothesis"] = hypothesis_translations
    data_dict["label"] = labels
    dataset = datasets.Dataset.from_dict(data_dict)
    return dataset


def custom_collate_fn(batch, tokenizer: T5Tokenizer) -> tuple:
    premise_inputs, hypothesis_inputs, labels = zip(*batch)
    premise_input_ids = tokenizer.batch_encode_plus(premise_inputs, padding=True, return_tensors='pt')['input_ids'].to(get_device())
    hypothesis_input_ids = tokenizer.batch_encode_plus(hypothesis_inputs, padding=True, return_tensors='pt')['input_ids'].to(get_device())
    labels = torch.tensor(labels).to(get_device())
    return premise_input_ids, hypothesis_input_ids, labels

if __name__ == "__main__":
    # Set Device
    device = get_device()
    
    # Load Data
    train, dev, test = get_data("snli")

    config = T5Config.from_pretrained("t5-small")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")  
    model.to(device)
    
    train_ds = SNLIDataset(train)
    dev_ds = SNLIDataset(dev)
    test_ds = SNLIDataset(test)
    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=False, collate_fn=lambda batch: custom_collate_fn(batch, tokenizer))
    dev_loader = DataLoader(dev_ds, batch_size=1024, shuffle=False, collate_fn=lambda batch: custom_collate_fn(batch, tokenizer))
    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False, collate_fn=lambda batch: custom_collate_fn(batch, tokenizer))
    
    # Save the dataset to disk
    dataset_cache_path: str = "./data/fnli"
    train_path = os.path.join(dataset_cache_path, "train_dataset")
    dev_path = os.path.join(dataset_cache_path, "dev_dataset")
    test_path = os.path.join(dataset_cache_path, "test_dataset")
    if not os.path.exists(dataset_cache_path):
        train_fnli = translate(train_loader, tokenizer, model, "Train")
        dev_fnli = translate(dev_loader, tokenizer, model, "Dev")
        test_fnli = translate(test_loader, tokenizer, model, "Test")
        train_fnli.save_to_disk(train_path)
        dev_fnli.save_to_disk(dev_path)
        test_fnli.save_to_disk(test_path)
    else:
        train_fnli = datasets.load_from_disk(train_path)
        dev_fnli = datasets.load_from_disk(dev_path)
        test_fnli = datasets.load_from_disk(test_path)