import gzip
import pandas as pd
from collections import Counter
from torch.utils.data import Dataset


def get_data():
    # Read clickbait headlines
    with gzip.open('dataset/clickbait_data.gz', 'rb') as f:
        clickbait_df = pd.read_csv(f, sep='\t', header=None, names=['headline'])
    clickbait_df['label'] = 1

    # Read non-clickbait headlines
    with gzip.open('dataset/non_clickbait_data.gz', 'rb') as f:
        non_clickbait_df = pd.read_csv(f, sep='\t', header=None, names=['headline'])
    non_clickbait_df['label'] = 0

    # Merge clickbait and non-clickbait dataframes
    data = pd.concat([clickbait_df, non_clickbait_df])

    # Shuffle the rows to randomize the order
    data = data.sample(frac=1, random_state=1).reset_index(drop=True)
    return data

class Vocabulary:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        self.tokenized_df = self.tokenize()
        self.unique_words = self.load_unique_words()
        self.word2idx = {word:idx for idx, word in enumerate(self.unique_words)}
        self.word2idx['<pad>'] = len(self.word2idx)
        self.word2idx['<unk>'] = len(self.word2idx)
        self.idx2word = {idx:word for word, idx in self.word2idx.items()}

    def load_unique_words(self) -> list:
        word_list = [word for _, line in self.df['tokenized'].items() for word in line]
        return Counter(word_list)
    
    def tokenize(self) -> pd.DataFrame:
        self.df['tokenized'] = self.df['headline'].apply(lambda line: ["<sos>"] + line.strip().split() + ["<eos>"])
        return self.df['tokenized']
    
    def __len__(self) -> int:
        return len(self.word2idx)
    
    
class ClickbaitDataset(Dataset):
    def __init__(self, df: pd.DataFrame, vocab: Vocabulary) -> None:
        self.df = df
        self.vocab = vocab
        self.tokenized_df = self.tokenize()
        self.word2idx = self.vocab.word2idx
        self.idx2word = self.vocab.idx2word
        
    def __getitem__(self, idx):
        line = self.tokenized_df.iloc[idx]
        text = [self.word2idx[word] if word in self.word2idx else self.word2idx['<unk>'] for word in line]
        label = self.df.iloc[idx]['label']
        return text, label
    
    def __len__(self):
        return len(self.df)
    
    def tokenize(self) -> pd.DataFrame:
        self.df['tokenized'] = self.df['headline'].apply(lambda line: ["<sos>"] + line.strip().split() + ["<eos>"])
        return self.df['tokenized']