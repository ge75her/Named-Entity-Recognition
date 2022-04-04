import torch
from torch.utils.data import Dataset, DataLoader


class NER(Dataset):
    def __init__(self, word_lists, tag_lists):
        self.word_lists = word_lists
        self.tag_lists = tag_lists

    def __getitem__(self, index):
        data = self.word_lists[index]
        tag = self.tag_lists[index]
        data = torch.tensor(data, dtype=torch.long)
        tag = torch.tensor(tag, dtype=torch.long)
        return data, tag

    def __len__(self):
        return len(self.word_lists)