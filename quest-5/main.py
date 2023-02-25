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
from models import LSTM, LSTM_With_Attention
from train import train
from evaluate import evaluate

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("--epochs", dest = "EPOCHS", type = int, default = 5)
argument_parser.add_argument("--seed", dest = "SEED", type = int, default = 42)
argument_parser.add_argument("--batch-size", dest = "BATCH_SIZE", type = int, default = 128)
argument_parser.add_argument("--embed-dim", dest = "EMBED_DIM", type = int, default = 100)
argument_parser.add_argument("--hidden-dim", dest = "HIDDEN_DIM", type = int, default = 256)
argument_parser.add_argument("--num-layers", dest = "NUM_LAYERS", type = int, default = 2)
argument_parser.add_argument("--bidirectional", dest = "BIDIRECTIONAL", type = bool, default = False)
argument_parser.add_argument("--optimizer", dest = "OPTIMIZER", type = str, default = 'Adam')
argument_parser.add_argument("--loss-fn", dest = "LOSS_FN", type = str, default = 'BCELoss')
argument_parser.add_argument("--score_-n", dest = "SCORE_FN", type = str, default = 'F1_Score')
argument_parser.add_argument("--learning-rate", dest = "LEARNING_RATE", type = float, default =  1e-3)
argument_parser.add_argument("--dropout", dest = "DROPOUT", type = float, default =  0.2)
argument_parser.add_argument("--heads", dest = "HEADS", type = int, default =  4)
argument_parser.add_argument("--model", dest = "MODEL", type = str, default = "LSTM")
args, _ = argument_parser.parse_known_args()


def epoch_time(start_time, end_time):
    """Calculate the time taken for an epoch."""
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == "__main__":
    # Set Device
    device = get_device()
    
    # Set Seed for Reproducibility
    make_reproducible(args.SEED)
    
    # Get Data from gzip files
    data = get_data()
    
    # Calculate and Print the percentage of clickbait headlines
    clickbait_percentage = (data[data["label"] == 1]["label"].count() / len(data)) * 100
    print(f"FLAG --> %age of clickbait headlines: {clickbait_percentage:.2f}%")
    
    # Split the data into Train, Dev and Test Sets
    train_data, dev_data = train_test_split(data, test_size=0.3, random_state=args.SEED)
    dev_data, test_data = train_test_split(dev_data, test_size=0.5, random_state=args.SEED)
    
    # Create Vocabulary from Training Data
    vocab = Vocabulary(train_data)
    
    # Input Dim is the size of the vocabulary
    INPUT_DIM = len(vocab)
    # Output Dim is 1 since we are doing binary classification
    OUTPUT_DIM = 1
    
    # Create Datasets
    train_ds = ClickbaitDataset(train_data, vocab)
    dev_ds = ClickbaitDataset(dev_data, vocab)
    test_ds = ClickbaitDataset(test_data, vocab)
    
    def custom_collate_fn(batch):
        """Custom collate function to pad the sequences to the maximum length in the batch."""
        texts, labels = zip(*batch)
        
        texts_tensor = [torch.tensor(text, device = device) for text in texts]
        labels_tensor = torch.tensor(labels, device = device)

        lengths = [len(text) for text in texts]
        lengths = torch.tensor(lengths, device = "cpu") # Lengths need to be on CPU
        
        texts_padded = pad_sequence(texts_tensor, batch_first = True, padding_value = vocab.word2idx['<pad>'])
        
        return texts_padded, labels_tensor, lengths
    
    # Create Dataloaders
    train_loader = DataLoader(train_ds, batch_size=args.BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=args.BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
    
    if args.MODEL == "LSTM":
        model = LSTM(INPUT_DIM, args.EMBED_DIM, args.HIDDEN_DIM, OUTPUT_DIM, args.NUM_LAYERS, args.BIDIRECTIONAL, args.DROPOUT).to(device)
    elif args.MODEL == "LSTM_With_Attention":
        model = LSTM_With_Attention(INPUT_DIM, args.EMBED_DIM, args.HIDDEN_DIM, OUTPUT_DIM, args.NUM_LAYERS, args.BIDIRECTIONAL, args.DROPOUT, args.HEADS).to(device)
    
    # Optimizer
    if args.OPTIMIZER == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr = args.LEARNING_RATE)

    # Loss Function
    if args.LOSS_FN == "BCELoss":
        loss_fn = nn.BCEWithLogitsLoss().to(device)

    # Initialize Best Validation Loss
    best_dev_loss = float("inf")
        
    # Path to Save Best Model
    PATH = f"lstm-best-model.pt"

    # Score Function
    if args.SCORE_FN == "F1_Score":
        score_fn = f1_score

    # Train and Evaluate the Model
    for epoch in range(args.EPOCHS):
        start_time = time.time()
        
        # Avg Train Loss, Accuracy
        train_loss, train_acc = train(train_loader, model, optimizer, loss_fn)

        # Avg Val Loss, F1_Score
        dev_loss, dev_f1 = evaluate(dev_loader, model, loss_fn, score_fn)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # Print Epoch Results Summary
        print(f"\n\tEpoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}")
        print(f"\tValidation Loss: {dev_loss:.3f} | F1_Score: {dev_f1:.3f}\n")

        if dev_loss < best_dev_loss:
            best_valid_loss = dev_loss
            torch.save(model.state_dict(), PATH)
            
    # Evaluation on Test Set
    model.load_state_dict(torch.load("lstm-best-model.pt"))
    model.eval()
    with torch.no_grad():
        test_loss, test_f1 = evaluate(test_loader, model, loss_fn, score_fn)
    print(f"FLAG --> Test Loss: {test_loss:.3f} | Test F1 Score: {test_f1:.3f}")