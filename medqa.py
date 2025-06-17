from datasets import load_dataset
import json

if __name__ == "__main__":
    # testing
    dataset = load_dataset("bigbio/med_qa") # I think all of these are english
    print(dataset)
    data = dataset["train"]
    print(json.dumps(data[0], indent=4))
    print('-' * 80)