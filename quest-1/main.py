import argparse
import time
import math
import torch
import torch.nn as nn
import torch.onnx
import torchinfo

import data
from utils import get_device, repackage_hidden, make_reproducible
from rnnlm import RNNModel

from collections import Counter
from torchtext.vocab import GloVe



def get_word_embedding(glove, word):
    """
    Returns the word embedding for a given word from a pre-trained GloVe model.
    
    Parameters:
    glove (object): A pre-trained GloVe model.
    word (str): The word for which the embedding is to be retrieved.
    
    Returns:
    torch.tensor: The word embedding for the given word.
    """
    return glove.vectors[glove.stoi[word]]


def init_glove_embeddings(model: RNNModel, freeze_embed, word2idx):
    """
    Initializes the word embeddings for the input layer of a given model using pre-trained GloVe embeddings.
    
    Parameters:
    model (RNNModel): The model for which the input layer embeddings are to be initialized.
    freeze_embed (bool): A flag to indicate whether to freeze the embedding layer or allow it to be updated during training.
    word2idx (dict): A dictionary mapping words to their index in the vocabulary.
    
    Returns:
    RNNModel: The input layer of the model, initialized with pre-trained GloVe embeddings.
    """
    # Load GloVe Embeddings
    GLOVE_DIM = 50
    glove = GloVe(name = '6B', 
                  dim = GLOVE_DIM)
    
    # Create Embeddings Matrix
    sos_embed = torch.rand(1, GLOVE_DIM) # Word Embedding for <sos>
    eos_embed = torch.rand(1, GLOVE_DIM) # Word Embedding for <eos>
    unk_embed = torch.rand(1, GLOVE_DIM) # Word Embedding for <unk>
    embeddings = list()
    for word, _ in word2idx.items():
        if word in glove.stoi: # If word present in GloVe
            embeddings.append(get_word_embedding(glove, word))
        else:
            if(word == '<sos>'): # If word is <sos>
                embeddings.append(sos_embed)
            elif(word == '<eos>'): # If word is <eos>
                embeddings.append(eos_embed)
            else: # If word is <unk> or not present in GloVe
                embeddings.append(unk_embed)
    temp_list = [emb.detach().cpu().numpy().squeeze().tolist() for emb in embeddings]
    # Tensor of Word Embeddings for each word in vocab
    embeddings = torch.tensor(temp_list, device = get_device())
        
    # Initialize Embedding Layer with Pre-Trained GloVe Embeddings
    model.in_embedder = nn.Embedding.from_pretrained(embeddings)
    
    # Freeze the Embedding Layer
    if freeze_embed:
        model.in_embedder.weight.requires_grad = False
    else:
        model.in_embedder.weight.requires_grad = True
    return model


def compute_perplexity(loss: float):
    """
    Compute perplexity of a language model.

    Args:
    loss (float): The cross-entropy loss calculated by the language model.

    Returns:
    float: The perplexity of the language model.
    """
    return math.exp(loss)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="./data/wikitext-2-mini",
        help="location of the data corpus",
    )

    parser.add_argument(
        "--emsize", type=int, default=50, help="size of word embeddings"
    )
    parser.add_argument(
        "--nhid", type=int, default=200, help="number of hidden units per layer"
    )
    parser.add_argument("--bidirectional", type=bool, default=True, help="bi-directional cell")
    parser.add_argument("--nlayers", type=int, default=2, help="number of layers")
    parser.add_argument("--lr", type=float, default=20, help="initial learning rate")
    parser.add_argument("--clip", type=float, default=0.25, help="gradient clipping")
    parser.add_argument("--epochs", type=int, default=40, help="upper epoch limit")
    parser.add_argument(
        "--batch_size", type=int, default=20, metavar="N", help="batch size"
    )
    parser.add_argument(
        "--freeze_embed", type=bool, default=False, help="freeze embeddings"
    )
    parser.add_argument(
        "--use-adam", type=bool, default=False, help="use adam optimizer"
    )
    parser.add_argument(
        "--use-glove", type=bool, default=False, help="use glove embeddings"
    )
    parser.add_argument("--seq-len", type=int, default=35, help="sequence length")
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="dropout applied to layers (0 = no dropout)",
    )
    parser.add_argument("--seed", type=int, default=1111, help="random seed")
    parser.add_argument("--rnn-type", type=str, default="elman", help="rnn type")
    parser.add_argument(
        "--log-interval", type=int, default=200, metavar="N", help="report interval"
    )
    parser.add_argument(
        "--save", type=str, default="model.pt", help="path to save the final model"
    )
    return parser.parse_args()


# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.


def batchify(data, batch_size):
    # Work out how cleanly we can divide the dataset into batch_size parts.
    n_batches = data.size(0) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, n_batches * batch_size)
    # Evenly divide the data across the batch_size batches.
    data = data.view(batch_size, -1).t().contiguous()
    return data.to(device)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.


def get_batch(source, i):
    seq_len = min(args.seq_len, len(source) - 1 - i)
    data = source[i : i + seq_len]
    target = source[i + 1 : i + 1 + seq_len].view(-1)
    return data, target


def evaluate(model, data_source, criterion):
    model.eval()
    total_loss = 0.0
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.seq_len):
            data, targets = get_batch(data_source, i)
            output, hidden = model(data, hidden)
            hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)


def train_model_step(corpus, args, model, criterion, epoch, lr):
    # Turn on training mode which enables dropout.
    train_data = batchify(corpus.train, args.batch_size)
    model.train()
    total_loss = 0.0
    start_time = time.time()
    ntokens = len(corpus.vocab)
    hidden = model.init_hidden(args.batch_size)
    if args.use_adam:
        optimizer =  torch.optim.Adam(model.parameters(), lr = lr)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.seq_len)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        if args.use_adam:
            optimizer.zero_grad()
        else:
            model.zero_grad()
        hidden = repackage_hidden(hidden)
        output, hidden = model(data, hidden)
        loss = criterion(output, targets)
        loss.backward()
        if args.use_adam:
            optimizer.step()
        else:
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            if args.use_glove:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                for p in model.parameters():
                    p.data.add_(p.grad, alpha=-lr)
            else:
                params_to_update = [p for p in model.parameters() if p.requires_grad == True]
                torch.nn.utils.clip_grad_norm_(params_to_update, args.clip)
                for p in params_to_update:
                    p.data.add_(p.grad, alpha=-lr)

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | "
                "loss {:5.2f} | ppl {:8.2f}".format(
                    epoch,
                    batch,
                    len(train_data) // args.seq_len,
                    lr,
                    elapsed * 1000 / args.log_interval,
                    cur_loss,
                    compute_perplexity(cur_loss),
                )
            )
            total_loss = 0
            start_time = time.time()
    # Loop over epochs.


def train_model(corpus, args, model, criterion):
    best_val_loss = None
    eval_batch_size = 10
    val_data = batchify(corpus.valid, eval_batch_size)
    lr = args.lr
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            train_model_step(corpus, args, model, criterion, epoch=epoch, lr=lr)
            val_loss = evaluate(model, val_data, criterion)
            print("-" * 89)
            print(
                "| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | "
                "valid ppl {:8.2f}".format(
                    epoch,
                    (time.time() - epoch_start_time),
                    val_loss,
                    compute_perplexity(val_loss),
                )
            )
            print("-" * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(args.save, "wb") as f:
                    torch.save(model, f)
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                lr /= 4.0
    except KeyboardInterrupt:
        print("-" * 89)
        print("Exiting from training early")


def test_model(corpus, args, model, criterion):  # Load the best saved model.
    model = RNNModel.load_model(args.save)
    eval_batch_size = 10
    test_data = batchify(corpus.test, eval_batch_size)
    test_loss = evaluate(model, test_data, criterion=criterion)
    print("=" * 89)
    print(
        "| End of training | test loss {:5.2f} | test ppl {:8.2f}".format(
            test_loss, math.exp(test_loss)
        )
    )
    print("=" * 89)
    
def percent_unk_tokens(corpus):
    """
    Calculate the percentage of unknown tokens in a corpus.

    Args:
    corpus (object): The corpus object, containing a vocab and training data.

    Returns:
    float: The percentage of unknown tokens in the corpus, rounded to 4 decimal places.
    """
    unk_token_idx = corpus.vocab.type2index['<unk>']
    print(f'idx of <unk> token: {unk_token_idx}')
    counts = Counter(corpus.train.detach().cpu().numpy())
    unk_count = counts[unk_token_idx]
    percentage_unk = unk_count/len(corpus.train) * 100
    return round(percentage_unk, 4)

if __name__ == "__main__":
    args = parse_args()
    make_reproducible(args.seed)
    device = get_device()
    print(args)
    
    corpus = data.Corpus(args.data)
    
    # Print % of <unk> tokens
    percentage_unk = percent_unk_tokens(corpus)
    print(f'% of <unk> tokens: {percentage_unk}')
            
    eval_batch_size = 10

    test_data = batchify(corpus.test, eval_batch_size)

    ntokens = len(corpus.vocab)
    
    model = RNNModel(ntokens,
                     args.emsize,
                     args.nhid,
                     args.nlayers,
                     args.bidirectional,
                     args.rnn_type,
                     args.dropout).to(device)
    
    # GloVe Embeddings
    if args.use_glove:
        model = init_glove_embeddings(model,
                                      args.freeze_embed, 
                                      corpus.vocab.type2index).to(device)
    
    # Model Summary
    print(model)
    torchinfo.summary(model)
    
    # Loss Function
    criterion = nn.NLLLoss()
    
    # Model Train Function 
    train_model(corpus, args, model, criterion)
    # Model Evaluate/Test Function 
    test_model(corpus, args, model, criterion)