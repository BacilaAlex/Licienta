import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.5):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers,dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, text):
        embedded = self.embedding(text)
        _ , (hidden, cell) = self.lstm(embedded)
        hidden = hidden.transpose(0, 1)
        hidden = torch.mean(hidden, dim=1)
        output = self.fc(hidden)
        return output
