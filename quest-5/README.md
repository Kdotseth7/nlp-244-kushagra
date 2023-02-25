# NLP 244 Quest-5 README

## Prerequisite

### Installation

```bash
pip install --upgrade torch argparse scikit-learn collections tqdm
```

## Usage

This command will run the code and print the results for all the FLAGS on the console with default arguments.

```bash
python main.py
```

This command will run the code with CLI arguments.

```bash
python main.py --epochs 10 --batch-size 256 --bidirectional true
```

Here,

- --epochs = number of epochs
- --batch_size = batch size
- --bidirectional = bidirectional lstm
- --dropout = dropout value
