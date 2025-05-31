import torch
import os

class ModelSaver:
    def __init__(self, model):
        self.model = model
        self.metrics = None
        self.bestEpoch = None
        self.modelPath = "Model/BestModel.pth"
        os.makedirs("Model", exist_ok=True)
    
    def SaveModel(self, metrics, epoch):
        accuracy = metrics.get('accuracy', 0)
        
        if self.metrics is None or accuracy > self.metrics.get('accuracy', 0):
            self.metrics = metrics.copy()
            self.bestEpoch = epoch
            
            torch.save({
                'ModelState': self.model.state_dict(),
                'Metrics': metrics,
                'Epoch': epoch
            }, self.modelPath)
            
            print(f"New best model saved at epoch {epoch}")
    
    
    def LoadModel(self, device):
        if not os.path.exists(self.modelPath):
            print(f"Warning: Model file not found at {self.modelPath}. Cannot load best model.")
        
        checkpoint = torch.load(self.modelPath, map_location=device)
        
        self.model.load_state_dict(checkpoint['ModelState'])
        self.model.to(device) 
        
        self.bestEpoch = checkpoint.get('Epoch')
        self.metrics = checkpoint.get('Metrics') 