from datasets import load_dataset

def load_msrvtt(split="train"):
    dataset = load_dataset("AlexZigma/msr-vtt")
    return dataset[split]

if __name__ == "__main__":
    data = load_msrvtt()
    print(data[0])
