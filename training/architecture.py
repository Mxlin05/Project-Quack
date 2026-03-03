import torch
from torch.utils.data import Dataset
import numpy as np

class TimeSeriesDataset(Dataset):
    def __init__(self, x, y, sequence_length):
        # Convert Pandas DataFrames to numpy arrays
        self.x = x.values if hasattr(x, 'values') else x
        self.y = y.values if hasattr(y, 'values') else y
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.x) - self.sequence_length

    def __getitem__(self, index):
        #Grab the features from the past amount of sequence length
        seq_x = self.x[index : index + self.sequence_length]
        
        #Grab the single target label at the END of that 60-minute window
        seq_y = self.y[index + self.sequence_length]
        
        #Convert to PyTorch tensors
        return torch.tensor(seq_x, dtype=torch.float32), torch.tensor(seq_y, dtype=torch.long)