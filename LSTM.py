import torch
import torch.nn as nn

# class LSTM(nn.Module):
#     def __init__(self, vocabularySize, outputSize, layers, hiddenSize):
#         super(LSTM, self).__init__()
        
#         # Create an embedding layer to convert token indices to dense vectors
#         self.embedding = nn.Embedding(vocabularySize, hiddenSize)
        
#         # Define the LSTM layer
#         self.lstm = nn.LSTM(input_size=hiddenSize, hidden_size=hiddenSize, num_layers=layers, batch_first=True)
        
#         # Define the output fully connected layer
#         self.fc_out = nn.Linear(hiddenSize, outputSize)

#     def forward(self, inputSeq, hiddenIn, memIn):
#         # Convert token indices to dense vectors
#         inputEmbs = self.embedding(inputSeq)

#         # Pass the embeddings through the LSTM layer
#         output, (hiddenOut, memOut) = self.lstm(inputEmbs, (hiddenIn, memIn))
                
#         # Pass the LSTM output through the fully connected layer to get the final output
#         return self.fc_out(output), hiddenOut, memOut

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers=1, dropout=0.5):
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
