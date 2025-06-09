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
        self.bestHarmonicMean = -1.0
        os.makedirs("Model", exist_ok=True)
    
    def SaveModel(self, epoch, metrics, trainingLosses, validationLosses, trainingAccuracies, validationAccuracies):
        f1Score = metrics.get('f1', 0.0)
        accuracy = metrics.get('accuracy', 0.0)
        
        if (accuracy + f1Score) == 0:
            harmonicMean = 0.0
        else:
            harmonicMean = 2 * (accuracy * f1Score) / (accuracy + f1Score)

        if harmonicMean > self.bestHarmonicMean:
            self.bestHarmonicMean = harmonicMean
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