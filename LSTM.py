import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.5):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers,dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, text):
        # text shape: [batch_size, seq_length]
        embedded = self.embedding(text)  # shape: [batch_size, seq_length, embedding_dim]
        
        _ , (hidden, cell) = self.lstm(embedded)
        # hidden shape: [num_layers, batch_size, hidden_dim]
        
        # Average hidden states from all layers
        # First, transpose to get [batch_size, num_layers, hidden_dim]
        hidden = hidden.transpose(0, 1)
        # Calculate mean across layers dimension (dim=1)
        hidden = torch.mean(hidden, dim=1)  # shape: [batch_size, hidden_dim]
        
        # Pass through linear layer
        output = self.fc(hidden)  # shape: [batch_size, num_classes]
        return output
