import torch
import os

class ModelSaver:
    def __init__(self, model, optimizer, scheduler):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = None
        self.bestEpoch = None
        self.modelPath = "Model/BestModel.pth"
        os.makedirs("Model", exist_ok=True)
    
    def SaveModel(self, epoch, metrics, trainingLosses, validationLosses, trainingAccuracies, validationAccuracies):
        accuracy = metrics.get('accuracy', 0)
        
        if self.metrics is None or accuracy > self.metrics.get('accuracy', 0):
            self.metrics = metrics.copy()
            self.bestEpoch = epoch
            
            torch.save({
                'Epoch': epoch,
                'ModelState': self.model.state_dict(),
                'Metrics': metrics,
                'Optimizer': self.optimizer.state_dict(),
                'Scheduler': self.scheduler.state_dict(),
                'TrainingLosses': trainingLosses,
                'ValidationLosses': validationLosses,
                'TrainingAccuracies': trainingAccuracies,
                'ValidationAccuracies': validationAccuracies
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