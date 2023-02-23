import torch
import os
import datasets
import optparse
from torch.utils.data import DataLoader
from utils import get_device
from data import get_data, SNLIDataset
from transformers import (T5Config, 
                          T5Tokenizer, 
                          T5ForConditionalGeneration, 
                          TrainingArguments, 
                          Trainer)
from translate import translate
from functools import partial
from datasets import DatasetDict


# Parse Command Line Arguments
optparser = optparse.OptionParser()
optparser.add_option("-b", "--batch-size", dest="batch_size", default=1024, type="int", help="Size of each batch")
optparser.add_option("-n", "--num-workers", dest="num_workers", default=0, type="int", help="Number of workers to use for dataloader")
optparser.add_option("-u", "--upload", dest="upload", default=False, action="store_true", help="Upload the dataset to HuggingFace")
(opts, _) = optparser.parse_args()


def custom_collate_fn(batch, tokenizer: T5Tokenizer) -> tuple:
    premise_inputs, hypothesis_inputs, labels = zip(*batch)
    premise_input_ids = tokenizer.batch_encode_plus(premise_inputs, max_length=512, padding=True, truncation=True, return_tensors='pt')['input_ids'].to(get_device())
    hypothesis_input_ids = tokenizer.batch_encode_plus(hypothesis_inputs, max_length=512, padding=True, truncation=True, return_tensors='pt')['input_ids'].to(get_device())
    labels = torch.tensor(labels).to(get_device())
    return premise_input_ids, hypothesis_input_ids, labels

if __name__ == "__main__":
    # Set Device
    device = get_device()
    
    # Set Multiprocessing
    if opts.num_workers > 0:
        torch.multiprocessing.set_start_method('spawn')
    
    # Load Data
    train, dev, test = get_data("snli")

    # Load Config, Tokenizer, Model
    config = T5Config.from_pretrained("t5-small")
    tokenizer = T5Tokenizer.from_pretrained("t5-small", model_max_length=512)
    model = T5ForConditionalGeneration.from_pretrained("t5-small")  
    model.to(device)
    
    # Create Datasets
    train_ds = SNLIDataset(train)
    dev_ds = SNLIDataset(dev)
    test_ds = SNLIDataset(test)
    
    # Create Dataloaders
    train_loader = DataLoader(train_ds, batch_size=opts.batch_size, num_workers=opts.num_workers, shuffle=False, collate_fn=partial(custom_collate_fn, tokenizer=tokenizer))
    dev_loader = DataLoader(dev_ds, batch_size=opts.batch_size, num_workers=opts.num_workers, shuffle=False, collate_fn=partial(custom_collate_fn, tokenizer=tokenizer))
    test_loader = DataLoader(test_ds, batch_size=opts.batch_size, num_workers=opts.num_workers, shuffle=False, collate_fn=partial(custom_collate_fn, tokenizer=tokenizer))
    
    # Save the dataset to disk
    dataset_cache_path: str = "./data/fnli"
    train_path = os.path.join(dataset_cache_path, "train_dataset")
    dev_path = os.path.join(dataset_cache_path, "dev_dataset")
    test_path = os.path.join(dataset_cache_path, "test_dataset")
    if not os.path.exists(dataset_cache_path):
        # Translate the dataset using T5
        with torch.no_grad():
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
    
    # API Token for HuggingFace
    os.environ["HF_API_TOKEN"] = "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"
    
    # Create DatasetDict for HuggingFace
    dataset_dict = DatasetDict({
        "train": train_fnli, 
        "dev": dev_fnli, 
        "test": test_fnli
    })
    
    # Upload the dataset to HuggingFace
    if opts.upload:
        dataset_dict.push_to_hub("snli-french", token=os.environ["HF_API_TOKEN"])