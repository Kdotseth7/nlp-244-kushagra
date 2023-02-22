from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from utils import get_device


class SNLIDataset(Dataset):
    def __init__(self, examples, tokenizer: T5Tokenizer):
        self.examples = examples
        self.tokenizer = tokenizer
        self.device = get_device()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index: int):
        example = self.examples[index]
        premise = example['premise']
        premise_input = f"Translate English to French: {premise}"
        hypothesis = example['hypothesis']
        hypothesis_input = f"Translate English to French: {hypothesis}"
        label = example['label']
        return premise_input, hypothesis_input, label