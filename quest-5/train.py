import torch
from tqdm import tqdm


def train(loader, model, optimizer, loss_fn) -> float:
    """Train the model on batches from the loader and return the loss and accuracy for the epoch."""
    model.train()
    losses = list()
    pbar = tqdm(loader, desc = "Training...", colour = 'red')
    for x, y, x_lengths in pbar:
        optimizer.zero_grad()
        
        # Calculate y_pred
        y_pred = model(x, x_lengths)
        
        loss = loss_fn(y_pred, y.float())
        pbar.set_postfix({"Loss": loss.item()})
        losses.append(loss.item())
        
        acc = binary_accuracy(y_pred, y)
        
        # Calculate gradients for w/b
        loss.backward()  
        # Update weights according to optimizer rules
        optimizer.step()          
    return sum(losses) / len(losses), acc

def binary_accuracy(preds, y):
    """Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8."""
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc