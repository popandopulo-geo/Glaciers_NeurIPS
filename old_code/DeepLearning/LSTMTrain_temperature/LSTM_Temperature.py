import torch.nn as nn
import torch
from torch.autograd import Variable

class LSTM_Temperature(nn.Module):
    def __init__(self, lstmLayers, lstmHiddenSize, lstmInputSize, dropout, device):
        super(LSTM_Temperature, self).__init__()

        self.lstmLayers = lstmLayers
        self.lstmHiddenSize = lstmHiddenSize
        self.lstmInputSize = lstmInputSize
        self.device = device

        # Define the LSTM layer for temperature data
        self.lstm = nn.LSTM(input_size=self.lstmInputSize, hidden_size=self.lstmHiddenSize, 
                            num_layers=self.lstmLayers, batch_first=True, dropout=dropout)

        # Linear layer to process LSTM output
        self.fc = nn.Linear(self.lstmHiddenSize, self.lstmInputSize)

    def forward(self, x):
        # x is the temperature data with shape (batch_size, sequence_length, 50, 50)
        
        # Flatten the spatial dimensions (50, 50) to a single dimension
        x = x.view(x.size(0), x.size(1), -1)  # Now x has shape (batch_size, sequence_length, 50*50)

        # Initialize hidden and cell states
        h_0 = Variable(torch.zeros(self.lstmLayers, x.size(0), self.lstmHiddenSize)).to(self.device)
        c_0 = Variable(torch.zeros(self.lstmLayers, x.size(0), self.lstmHiddenSize)).to(self.device)

        # Propagate input through LSTM
        lstm_out, _ = self.lstm(x, (h_0, c_0))

        # Pass through a linear layer to reduce dimensions
        out = self.fc(lstm_out[:, -1, :])  # Take the last output

        return out  # Shape: (batch_size, lstmInputSize)

"""
# Example usage:
device = "cuda" if torch.cuda.is_available() else "cpu"
model = LSTM_Temperature(1, 32, 50*50, 0.1, device).to(device)
temp_data = torch.rand(1, 30, 50,50).to(device)  # (batch_size, sequence_length, 50*50)
output = model(temp_data)
print(output.shape)  # (batch_size, 50*50)
"""