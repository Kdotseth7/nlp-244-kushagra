from utils import get_device
from data import get_data
from transformers import T5Tokenizer, T5ForConditionalGeneration


def translate(text, model):
    # Tokenize the text
    inputs = tokenizer.encode("translate English to French: " + text, return_tensors="pt").to(get_device())
    # Generate the translation
    outputs = model.generate(inputs)
    # Decode the translation
    translated_text = tokenizer.decode(outputs[0])
    # Remove the "translate French to English:" prefix and any leading/trailing white space
    return translated_text.replace("translate English to French: ","").strip()

if __name__ == "__main__":
    # Set Device
    device = get_device()
    print(device)
    
    # Load Data
    train, dev, test = get_data()
    print(train[0])

    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")  
    model.to(device)
    
    j=0
    for i in train:
        print(translate(i["premise"], model))
        print(translate(i["hypothesis"], model))
        print(i["label"])
        print("=====================================")
        j+=1
        if j==10:
            break