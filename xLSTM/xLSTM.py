import torch.nn as nn
from .mLSTMblock import mLSTMblock
from .sLSTMblock import sLSTMblock

class xLSTM(nn.Module):
    def __init__(self, layers, batchPaddingInputSize, depth=4, factor=2):
        print(f"[INFO] Initializing xLSTM with layers: {layers}")
        super(xLSTM, self).__init__()

        self.layers = nn.ModuleList()
        for layer_type in layers:
            if layer_type == 's':
                layer = sLSTMblock(batchPaddingInputSize, depth)
            elif layer_type == 'm':
                layer = mLSTMblock(batchPaddingInputSize, factor, depth)
            else:
                raise ValueError(f"Invalid layer type: {layer_type}. Choose 's' for sLSTM or 'm' for mLSTM.")
            self.layers.append(layer)
        print("[INFO] xLSTM layers initialized.")
    
    def init_states(self, x):
        [l.init_states(x) for l in self.layers]
        
    def forward(self, x):
        x_original = x.clone()
        for l in self.layers:
            x = l(x) + x_original
        return x