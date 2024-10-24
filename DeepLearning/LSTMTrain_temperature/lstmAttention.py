import torch.nn as nn
import torch
from torch.autograd import Variable
import functions
import math

class TemperatureLSTM(nn.Module):
    def __init__(self, lstmInputSize, lstmHiddenSize, num_layers, device):
        super(TemperatureLSTM, self).__init__()
        self.device = device
        self.num_layers = num_layers
        self.lstmHiddenSize = lstmHiddenSize
        self.lstmInputSize = lstmInputSize
        self.lstm = nn.LSTM(input_size=self.lstmInputSize, hidden_size=self.lstmHiddenSize, 
                            num_layers=num_layers, batch_first=True)
        
    def forward(self, temperatures):
        # temperatures is a list of 4 tensors of shape (X, 50, 50)
        h = Variable(torch.zeros(self.num_layers, 1, self.lstmHiddenSize)).to(self.device)  # hidden state
        c = Variable(torch.zeros(self.num_layers, 1, self.lstmHiddenSize)).to(self.device)  # cell state
        outputs = []
        for temp in temperatures:
            temp = temp.view(temp.size(1), temp.size(2) * temp.size(3))  # Flatten 50x50 to (X, 2500)
            output, (h, c) = self.lstm(temp.unsqueeze(0), (h, c))  # LSTM expects input of shape (batch_size, seq_len, input_size)
            outputs.append(output[:, -1, :])  # We only need the last output

        # Concatenate the outputs from all temperature tensors
        lstm_output = torch.stack(outputs, dim=1)  # Shape: (batch_size, 4, hidden_size)

        return lstm_output


class LSTM(nn.Module):
    def __init__(self, lstmLayersEnc, lstmLayersDec, lstmHiddenSize, lstmInputSize, dropout, attentionHeads, device):
        super(LSTM, self).__init__()
        # global
        self.flatten = nn.Flatten(start_dim = 2, end_dim = 3)

        # LSTM layers encoder
        self.device = device
        self.dropout = dropout
        self.lstmLayersEnc = lstmLayersEnc
        self.lstmLayersDec = lstmLayersDec
        self.lstmHiddenSize = lstmHiddenSize
        self.lstmInputSize = lstmInputSize
        self.lstmEncoder = nn.LSTM(input_size=self.lstmInputSize, hidden_size=self.lstmHiddenSize,
                            num_layers=self.lstmLayersEnc, batch_first=True, dropout = self.dropout)

        # decoder
        self.flattenDec = nn.Flatten(start_dim=1, end_dim=2)
        self.attention = nn.MultiheadAttention(self.lstmInputSize, attentionHeads, self.dropout, batch_first=True)
        self.linear = nn.Sequential(nn.Linear(12500, 2500))

        # Add the Temperature 
        self.temperature_lstm = TemperatureLSTM(self.lstmInputSize,self.lstmHiddenSize, 1, device)

    def encoder(self, x):
        """
        encodes the input with LSTM cells

        x: torch.tensor
            (b, s, dim)
        return list of torch.tensor and tuple of torch.tensor and torch.tensor
            output, hidden and cell state
        """

        # init hidden and cell state
        h_0 = Variable(torch.zeros(self.lstmLayersEnc, x.size(0), self.lstmHiddenSize)).to(self.device)  # hidden state
        c_0 = Variable(torch.zeros(self.lstmLayersEnc, x.size(0), self.lstmHiddenSize)).to(self.device)  # internal state

        # Propagate input through LSTM
        output, _ = self.lstmEncoder(x, (h_0, c_0))  # lstm with input, hidden, and internal state

        return output

    def decoder(self, outputEnc, temperatures, y, training):
        """
        applies MHA to output of LSTM encoder

        outputEnc: torch.tensor
        y: torch.tensor
        training: boolean

        returns: torch.tensor

        """
        temp_lstm_output = self.temperature_lstm(temperatures)

        print(temp_lstm_output.shape)
        
        if training == False:
            out = []
            for i in range(4):
                x, _ = self.attention(outputEnc, outputEnc, outputEnc)
                #print("------------------------------------------")
                #print(i, x.shape, outputEnc.shape)
                x = self.flattenDec(x)
                #print('self.flattenDec(x): ', x.shape)
                x = torch.cat((x, temp_lstm_output[:,i,:]), dim=1) 
                #print('x.shape', x.shape)
                x = self.linear(x)
                #print('self.linear(x)', x.shape)
                out.append(x)

                # update hidden input
                outputEnc = torch.cat((outputEnc, x.unsqueeze(dim=1)), dim=1)  # teacher forcing
                outputEnc = self.encoder(outputEnc[:, -4:, :])  # take last 4 inputs

        if training == True:
            out = []
            for i in range(4):
                x, _ = self.attention(outputEnc, outputEnc, outputEnc)
                #print(i, x.shape, outputEnc.shape)
                x = self.flattenDec(x)
                x = torch.cat((x, temp_lstm_output[:,i,:]), dim=1) 
                x = self.linear(x)
                out.append(x)
                
                # update hidden input
                outputEnc = torch.cat((outputEnc, y[:,i,:].unsqueeze(dim =1)), dim=1) # teacher forcing
                outputEnc = self.encoder(outputEnc[:, -4:, :])  # take last 4 inputs


        out = torch.stack(out, dim = 1)
        out = out.reshape(out.size(dim = 0), out.size(dim = 1), 50, 50)

        return out

    def forward(self, x, temperatures, y, training):

        # flatten input
        x = self.flatten(x)
        s = self.encoder(x)
        y = self.flatten(y)
        output = self.decoder(s, temperatures, y, training)

        return output


"""# test, args: lstmLayersEnc, lstmLayersDec, lstmHiddenSize, lstmInputSize, dropout, attentionHeads, device
device = "cpu"
model = LSTM(1,1, 2500, 2500, 0.1, 1, device).to(device)
Temperatures = [torch.rand(12, 50,50).to(device),torch.rand(11, 50,50).to(device),
                torch.rand(13, 50,50).to(device),torch.rand(10, 50,50).to(device)]
test = torch.rand(1, 4, 50,50).to(device)

print(model(test,Temperatures, test, training = False).size())"""







