# NLP 244 Quest-5 README

## Prerequisite

### Installation

```bash
pip install --upgrade torch argparse scikit-learn collections tqdm
```

## Usage

This command will run the LSTM Model and print the results for all the FLAGS on the console with default arguments.

```bash
python main.py --model LSTM
```

This command will run the LSTM With Attention Model and print the results for all the FLAGS on the console with default arguments.

```bash
python main.py --model LSTM_With_Attention
```

You can also run either models with CLI arguments.

```bash
python main.py --model LSTM --epochs 10 --batch-size 256 --bidirectional true
```

You can use the below-mentioned CLI arguments,

- --epochs = number of epochs
- --batch_size = batch size
- --learning-rate = learning rate
- --bidirectional = bidirectional lstm
- --embed-dim = embedding dimension
- --num-layers = number of LSTM layers
- --hidden-dim = hidden dimension for LSTM
- --dropout = dropout value
- --heads = number of attention heads if using LSTM_With_Attention model
