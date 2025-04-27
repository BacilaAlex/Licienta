import torch
import torch.nn as nn
from .xLSTM import xLSTM

class xLSTMClassifier(nn.Module):
    def __init__(self, layers, batchSize, vocabularySize, inputSize, depth, factor, dropout_head):
        super().__init__()
        self.embedding = nn.Embedding(vocabularySize, inputSize)
        batchSequenceInputSize = torch.zeros((batchSize, vocabularySize, inputSize))
        self.xLSTM = xLSTM(layers, batchSequenceInputSize, depth=depth, factor=factor)

        # 2) A little dropout, then linear map to num_classes
        self.dropout = nn.Dropout(dropout_head)
        self.classifier = nn.Linear(inputSize, 1)

    def forward(self, x):
        # x:  [batch, seq_len, feature_dim]
        emb = self.embedding(x)
        # a) reset recurrent states from the new batch
        self.xLSTM.init_states(emb)
        # b) run x through xLSTM
        seq_out = self.xLSTM(emb)  
        #    -> shape [batch, seq_len, feature_dim]

        # c) pool the time dimension—here we take the last time‑step
        #    you could also do mean: seq_out.mean(dim=1)
        rep = seq_out[:, -1, :]     # shape [batch, feature_dim]

        # d) classification head
        rep = self.dropout(rep)
        logits = self.classifier(rep).squeeze(-1) # shape [batch, num_classes]

        return logits
