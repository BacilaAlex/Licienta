import torch
import torch.nn as nn
from .xLSTM import xLSTM

class xLSTMClassifier(nn.Module):
    def __init__(self, layers, batchSize, vocabularySize, inputSize, depth, factor, dropout_head):
        super().__init__()
        self.embedding = nn.Embedding(vocabularySize, inputSize)
        batchSequenceInputSize = torch.zeros((batchSize, vocabularySize, inputSize))
        self.xLSTM = xLSTM(layers, batchSequenceInputSize, depth=depth, factor=factor)
        self.dropout = nn.Dropout(dropout_head)
        self.classifier = nn.Linear(inputSize, 1)

    def forward(self, x):
        emb = self.embedding(x)
        self.xLSTM.init_states(emb)
        seq_out = self.xLSTM(emb)  
        rep = seq_out[:, -1, :]
        rep = self.dropout(rep)
        logits = self.classifier(rep).squeeze(-1)
        return logits
