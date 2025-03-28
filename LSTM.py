
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, vocabularySize, outputSize, layers, hiddenSize):
        super(LSTM, self).__init__()
        
        # Create an embedding layer to convert token indices to dense vectors
        self.embedding = nn.Embedding(vocabularySize, hiddenSize)
        
        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size=hiddenSize, hidden_size=hiddenSize, num_layers=layers, batch_first=True, dropout=0.5)
        
        # Define the output fully connected layer
        self.fc_out = nn.Linear(hiddenSize, outputSize)

    def forward(self, input_seq, hidden_in, mem_in):
        # Convert token indices to dense vectors
        input_embs = self.embedding(input_seq)

        # Pass the embeddings through the LSTM layer
        output, (hidden_out, mem_out) = self.lstm(input_embs, (hidden_in, mem_in))
                
        # Pass the LSTM output through the fully connected layer to get the final output
        return self.fc_out(output), hidden_out, mem_out