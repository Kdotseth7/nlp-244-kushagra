import torch
import os
import datasets
import optparse
import wandb as wandb
import numpy as np
from torch.utils.data import DataLoader
from utils import get_device
from data import get_data, SNLIDataset
from transformers import (T5Config, 
                          T5Tokenizer, 
                          T5ForConditionalGeneration, 
                          TrainingArguments, 
                          AutoConfig, 
                          AutoTokenizer, 
                          AutoModelForSequenceClassification, 
                          Trainer, 
                          EvalPrediction)
from translate import translate
from functools import partial
from datasets import DatasetDict
from sklearn.metrics import f1_score


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


def tokenize(batch, tokenizer: AutoTokenizer):
    return tokenizer(batch["premise"], batch["hypothesis"], padding=True, truncation=True, max_length=512, return_tensors="pt")


def my_compute_metrics(eval_pred: EvalPrediction) -> dict:
    predictions, labels = eval_pred.predictions, eval_pred.label_ids
    predictions = np.argmax(predictions, axis=1)
    acc = (predictions == labels).astype(float).mean()
    f1 = f1_score(labels, predictions, average='weighted')
    return {"accuracy": acc.item(), "f1": f1}


if __name__ == "__main__":
    # Set Device
    device = get_device()
    
    # Set Multiprocessing
    if opts.num_workers > 0:
        torch.multiprocessing.set_start_method('spawn')
    
    # Load Data
    train, dev, test = get_data("snli")

    # Load Config, Tokenizer, Model for T5
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
        
    # Get the uploaded dataset
    fnli_dataset: DatasetDict = datasets.load_dataset("kseth919/snli-french", cache_dir="./data/cache/huggingface/datasets")
    
    # FNLI Dataset
    train_fnli = fnli_dataset["train"]
    dev_fnli = fnli_dataset["dev"]
    test_fnli = fnli_dataset["test"]
        
    # Load the Config, Tokenizer, and Model for CamemBERT
    cb_config = AutoConfig.from_pretrained('cmarkea/distilcamembert-base')   
    cb_tokenizer = AutoTokenizer.from_pretrained('cmarkea/distilcamembert-base')
    cb_model = AutoModelForSequenceClassification.from_pretrained("cmarkea/distilcamembert-base", num_labels=3)
    cb_model.to(device)
    
    BATCH_SIZE = 128
    
    train_fnli = train_fnli.map(partial(tokenize, tokenizer=cb_tokenizer), batched=True, batch_size=BATCH_SIZE)
    dev_fnli = dev_fnli.map(partial(tokenize, tokenizer=cb_tokenizer), batched=True, batch_size=BATCH_SIZE)
    test_fnli = test_fnli.map(partial(tokenize, tokenizer=cb_tokenizer), batched=True, batch_size=BATCH_SIZE)
    
    # Let's fine-tune with the Trainer API!
    training_args: TrainingArguments = TrainingArguments(
        output_dir="./data/models",
        do_train=True,
        do_eval=True,
        do_predict=True,
        evaluation_strategy="steps",
        eval_steps=128,
        per_device_train_batch_size=512,
        per_device_eval_batch_size=1024,
        save_steps=128,
        save_strategy="steps",
        save_total_limit=5,
        report_to=["wandb"],
        logging_steps=50,
        num_train_epochs=3,
        metric_for_best_model="accuracy",
        load_best_model_at_end=True,
        dataloader_num_workers=8,  # set to 0 when debugging and >1 when running!
    )
    
    # API Token for WANDB
    os.environ["WANDB_API_TOKEN"] = "c5c4a689c34310341a891f54e7b875dca6da6e42"
    wandb.login(key=os.environ["WANDB_API_TOKEN"])
    wandb.init(entity="kushagraseth-1996", project="nlp244", group="finetune_w_trainer")
    
    # Create TrainingArguments    
    trainer: Trainer = Trainer(
        model=cb_model,
        args=training_args,
        data_collator=None,  # let HF set this to an instance of transformers.DataCollatorWithPadding
        train_dataset=train_fnli,
        eval_dataset=dev_fnli,
        tokenizer=cb_tokenizer,
        compute_metrics=my_compute_metrics,
    )
    
    # Train the model
    trainer.train()
    model = trainer.model # make sure to load_best_model_at_end=True!
    
    # Run a final evaluation on the test set
    trainer.evaluate(metric_key_prefix="test", eval_dataset=test_fnli)