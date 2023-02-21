import utils

from datasets import load_dataset

if __name__ == "__main__":
    dataset =  load_dataset("snli")
    print(dataset["validation"][0])
    
    print(utils.get_device())