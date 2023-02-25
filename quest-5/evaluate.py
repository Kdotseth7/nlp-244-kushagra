from tqdm import tqdm
import torch

# Model Evaluate Function
def evaluate(loader, 
             model, 
             loss_fn, 
             score_fn):
    model.eval()
    losses = list()
    pbar = tqdm(loader, desc = 'Evaluation...', colour = 'green')
    for x, y, x_lengths in pbar:

        # Calculate y_pred
        y_pred = model(x, x_lengths)
        
        loss = loss_fn(y_pred, y.float())
        pbar.set_postfix({'Loss': loss.item()})
        losses.append(loss.item())
        
        y = y.detach().cpu().numpy()
        y_pred = torch.round(torch.sigmoid(y_pred))
        y_pred = y_pred.detach().cpu().numpy()
        score = score_fn(y, y_pred)
              
    return sum(losses) / len(losses), score