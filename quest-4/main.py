import torch
import pandas as pd
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


def translate(loader: DataLoader, tokenizer: T5Tokenizer, model: T5ForConditionalGeneration) -> list:
    data_dict = dict()
    premise_translations = []
    hypothesis_translations = []
    labels = []
    pbar = tqdm(loader, desc = 'Translating using T5...', colour = 'green')
    for premise, hypothesis, label in pbar:
        # # Generate the translation
        # outputs = model.generate(inputs)
        # # Decode the translation
        # translated_text = tokenizer.decode(outputs[0])
        # # Remove the "Translate French to English:" prefix and any leading/trailing white space
        # return translated_text.replace("Translate English to French: ","").strip()
        premise_output_batch = model.generate(premise)
        premise_translations.extend(tokenizer.batch_decode(premise_output_batch, skip_special_tokens=True))
        hypothesis_output_batch = model.generate(hypothesis)
        hypothesis_translations.extend(tokenizer.batch_decode(hypothesis_output_batch, skip_special_tokens=True))
        labels.extend(label.tolist())
    data_dict["premise"] = premise_translations
    data_dict["hypothesis"] = hypothesis_translations
    data_dict["label"] = labels
    df = pd.DataFrame(data_dict)
    return df


def custom_collate_fn(batch) -> tuple:
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
    
    train_ds = SNLIDataset(train, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=False, collate_fn=custom_collate_fn)
    
    df = translate(train_loader, tokenizer, model)
    df.head(5)