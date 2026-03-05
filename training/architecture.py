import torch
import torch.nn as nn
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
        
        #Grab the single target label at the end of sequence length
        seq_y = self.y[index + self.sequence_length]
        
        #Convert to PyTorch tensors
        return torch.tensor(seq_x, dtype=torch.float32), torch.tensor(seq_y, dtype=torch.long)

class LSTMModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, batch_first):
        super().__init__(self)

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first)
        self.linear = nn.Linear(input_size, output_size)
    

    def forward_pass(self, x):
        output, _ = self.lstm(x)
        output = self.linear(x)
        return output
    
class PositiveEncodingModel(nn.Module):
    def __init__(self, num_embeddings, output_dim):
        super().__init__(self)
        self.embedding = nn.Embedding(num_embeddings, output_dim)

    def forward_pass(self, minute_tensor, day_tensor):
        min_output = self.embedding(minute_tensor)
        day_output = self.embedding(day_tensor)

        return min_output, day_output
    
class TransformerModel(nn.Module):
    def __init__(self, input_size, output_size, d_model, n_head, batch_first ):
        super().__init__(self)

        self.linear = nn.Linear(input_size,output_size)
        self.transformer = nn.Transformer(d_model, n_head, batch_first=batch_first)

    def forward_pass(self, x):
        output = self.linear(x)
        output = self.transformer(x)
        output = self.linear(x)
        return output

