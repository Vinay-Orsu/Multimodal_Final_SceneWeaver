from datasets import load_dataset

def get_dataset(name: str, split: str = "train", streaming: bool = False):
    """
    Caption-first dataset loading.
    streaming=True avoids downloading full dataset locally.
    """
    name = name.lower()

    if name in ["msrvtt", "msr-vtt", "msr vtt"]:
        return load_dataset("AlexZigma/msr-vtt", split=split, streaming=streaming)

    raise ValueError(f"Unknown dataset: {name}")


if __name__ == "__main__":
    ds = get_dataset("msrvtt", split="train", streaming=False)
    first = ds[0] if hasattr(ds, "__getitem__") else next(iter(ds))
    print(first)
