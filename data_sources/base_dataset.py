class BaseDataset:
    def __init__(self, split="train"):
        self.split = split

    def load(self):
        raise NotImplementedError

    def get_prompt(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
