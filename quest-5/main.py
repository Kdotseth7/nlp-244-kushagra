import argparse
import torch
import torch.nn as nn
import time
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score
from utils import get_device, make_reproducible
from data import ClickbaitDataset, Vocabulary, get_data
from models import LSTM
from train import train
from evaluate import evaluate

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("--epochs", dest = "EPOCHS", type = int, default = 5)
argument_parser.add_argument("--seed", dest = "SEED", type = int, default = 42)
argument_parser.add_argument("--batch_size", dest = "BATCH_SIZE", type = int, default = 128)
argument_parser.add_argument("--embed_dim", dest = "EMBED_DIM", type = int, default = 100)
argument_parser.add_argument("--hidden_dim", dest = "HIDDEN_DIM", type = int, default = 256)
argument_parser.add_argument("--num_layers", dest = "NUM_LAYERS", type = int, default = 2)
argument_parser.add_argument("--bidirectional", dest = "BIDIRECTIONAL", type = bool, default = True)
argument_parser.add_argument("--optimizer", dest = "OPTIMIZER", type = str, default = 'AdamW')
argument_parser.add_argument("--loss_fn", dest = "LOSS_FN", type = str, default = 'BCELoss')
argument_parser.add_argument("--score_fn", dest = "SCORE_FN", type = str, default = 'F1_Score')
argument_parser.add_argument("--learning_rate", dest = "LEARNING_RATE", type = float, default =  1e-3)
argument_parser.add_argument("--dropout", dest = "DROPOUT", type = float, default =  0.2)
args, _ = argument_parser.parse_known_args()


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == "__main__":
    # Set Device
    device = get_device()
    
    # Set Seed
    make_reproducible(args.SEED)
    
    # Calculate the percentage of clickbait headlines
    data = get_data()
    
    clickbait_percentage = (data[data['label'] == 1]['label'].count() / len(data)) * 100

    # Print the percentage of clickbait headlines
    print(f"Percentage of clickbait headlines: {clickbait_percentage:.2f}%")
    
    train_data, dev_data = train_test_split(data, test_size=0.3, random_state=args.SEED)
    dev_data, test_data = train_test_split(dev_data, test_size=0.5, random_state=args.SEED)
    print(f"Train size: {len(train_data)}")
    print(f"Dev size: {len(dev_data)}")
    print(f"Test size: {len(test_data)}")
    
    vocab = Vocabulary(train_data)
    
    argument_parser.add_argument("--input_dim", dest = "INPUT_DIM", type = int, default = len(vocab.word2idx))
    argument_parser.add_argument("--output_dim", dest = "OUTPUT_DIM", type = int, default = 1)
    args, _ = argument_parser.parse_known_args()
    
    train_ds = ClickbaitDataset(train_data, vocab)
    dev_ds = ClickbaitDataset(dev_data, vocab)
    test_ds = ClickbaitDataset(test_data, vocab)
    
    def custom_collate_fn(batch):
        texts, labels = zip(*batch)
        
        texts_tensor = [torch.tensor(text, device = device) for text in texts]
        labels_tensor = torch.tensor(labels, device = device)

        lengths = [len(text) for text in texts]
        lengths = torch.tensor(lengths, device = 'cpu') # Lengths need to be on CPU
        
        texts_padded = pad_sequence(texts_tensor, batch_first = False, padding_value = vocab.word2idx['<pad>'])
        
        return texts_padded, labels_tensor, lengths
    
    train_loader = DataLoader(train_ds, batch_size=args.BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=args.BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
    
    model = LSTM(args.INPUT_DIM, 
             args.EMBED_DIM, 
             args.HIDDEN_DIM, 
             args.OUTPUT_DIM, 
             args.NUM_LAYERS, 
             args.BIDIRECTIONAL, 
             args.DROPOUT).to(device)

    print('LSTM Model: ', model)
    
    # Optimizer
    if args.OPTIMIZER == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), 
                                    lr = args.LEARNING_RATE)

    # Loss Function
    if args.LOSS_FN == 'BCELoss':
        loss_fn = nn.BCEWithLogitsLoss().to(device)

    # Initialize Best Validation Loss
    best_valid_loss = float('inf')
        
    # Path to Save Best Model
    PATH = f'lstm-best-model.pt'

    # Score Function
    if args.SCORE_FN == 'F1_Score':
        score_fn = f1_score

    for epoch in range(args.EPOCHS):

        start_time = time.time()
        
        # Avg Train Loss, Train Accuracy
        train_loss = train(train_loader, 
                           model, 
                           optimizer, 
                           loss_fn)

        # Avg Val Loss, F1_Score
        val_loss, val_acc = evaluate(dev_loader, 
                                     model, 
                                     loss_fn, 
                                     score_fn)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'\n\tEpoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\tValidation Loss: {val_loss:.3f} | F1_Score: {val_acc*100:.2f}%\n')

        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            torch.save(model.state_dict(), PATH)