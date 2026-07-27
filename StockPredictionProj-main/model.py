import torch
import torch.nn as nn
import torch.nn.functional as F
import dataManip
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout, batch_norm = True, norm = True):
        super(LSTMModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.norm = norm
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        

    def forward(self, x):
        if self.norm == True:
            # std, mean = torch.std_mean(x, dim=1)
            # x = (x-mean.unsqueeze(1))/std.unsqueeze(1)
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device)
            
            out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
            out = self.fc(out[:,-1,:])
            # out = (out*std)+mean
            return out
        
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device)

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:,-1,:])


        return out

