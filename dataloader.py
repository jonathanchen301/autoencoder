import torch
from torch.utils.data import Dataset

class DataFrameDataset(Dataset):

    def __init__(self, dataframe, transform=None):
        self.data = dataframe
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx].values
        row = torch.tensor(row).float()
        if self.transform:
            row = self.transform(row)
        return row