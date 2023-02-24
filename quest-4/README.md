# NLP 244 Quest-4 README

## Prerequisite

### Installation

```bash
pip install --upgrade torch transformers datasets sentencepiece wandb tqdm
```

## Usage

```bash
python main.py -n 8 -e 3 -u
```

This command will create the snli-french dataset and perform the fine-tuning using DistilCamemBERT

here,

- n = number of workers to use for dataloader
- e = number of epochs to train the model
- u = upload the dataset to HuggingFace

## Weights & Biases URL

```bash
https://wandb.ai/kushagraseth-1996/nlp244/groups/finetune_w_trainer?workspace=user-kushagraseth-1996
```

## Translated French SNLI Dataset on HuggingFace Hub

```bash
https://huggingface.co/datasets/kseth919/snli-french/viewer/kseth919--snli-french
```
