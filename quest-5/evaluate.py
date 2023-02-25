from tqdm import tqdm


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
        y_pred = model(x, x_lengths).squeeze(1)
        
        loss = loss_fn(y_pred, y.float())
        pbar.set_postfix({'Loss': loss.item()})
        losses.append(loss.item())

        score = score_fn(y, y_pred)
              
    return sum(losses) / len(losses), score