from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
from datasets import Dataset


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
    dataset = Dataset.from_dict(data_dict)
    return dataset