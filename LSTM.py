import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers=1):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        
        # If no hidden state is provided, initialize to zeros
        if hidden is None:
            batch_size = x.size(0)
            hidden = (torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.device), torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.device))
        
        lstm_out, hidden = self.lstm(embedded, hidden)  # Keep hidden state
        
        last_hidden = lstm_out[:, -1, :]
        logits = self.fc(last_hidden)
        
        return logits, hidden  # Return hidden state for next batch
