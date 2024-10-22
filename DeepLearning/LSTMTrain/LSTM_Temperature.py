import torch
import torch.nn as nn

class LSTMTemperature(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, device, dropout=0.1):
        super(LSTMTemperature, self).__init__()

        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                            num_layers=num_layers, batch_first=True, dropout=dropout)
        
        # Fully connected layer to map the LSTM output to desired size (reduce or expand as needed)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden and cell state
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h_0, c_0))  

        # Pass through fully connected layer
        out = self.fc(out[:, -1, :])  # Take only the last output for final prediction

        return out
