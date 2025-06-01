import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, recall_score, precision_score)
import os

class PlotGenerator:
    def __init__(self, device, model, testData, outputDir="Graphs"):
        self.device = device
        self.model = model
        self.testData = testData
        self.outputDir = outputDir
        os.makedirs(self.outputDir, exist_ok=True)
    
    def Generate(self, trainLosses, validationLosses, trainAccuracies, validationAccuracies):
        predictions, actuals = self.GetPredictions()
        
        predictions = [int(p) for p in predictions]
        actuals = [int(a) for a in actuals] 

        metrics = self.CalculateMetrics(actuals, predictions)

        self.TrainingHistoryPlot(trainLosses, validationLosses, trainAccuracies, validationAccuracies)
        self.ConfusionMatrixPlot(actuals, predictions)
        self.MetricsComparisonPlot(metrics)

    def GetPredictions(self):
        self.model.eval()
        predictions, actuals = [], []
        
        with torch.no_grad():
            for data, target in self.testData:
                data, target = data.to(self.device), target.to(self.device)
                target = target.view(-1, 1)
                
                output = self.model(data)
                prediction = (torch.sigmoid(output) > 0.5).float()
                
                predictions.extend(prediction.squeeze().cpu().numpy().tolist())
                actuals.extend(target.squeeze().cpu().numpy().tolist())
        
        return predictions, actuals
    
    def CalculateMetrics(self, actuals, predictions):
        return {
            'accuracy': accuracy_score(actuals, predictions),
            'f1': f1_score(actuals, predictions, zero_division=0),
            'recall': recall_score(actuals, predictions, zero_division=0),
            'precision': precision_score(actuals, predictions, zero_division=0)
        }
    
    def TrainingHistoryPlot(self, trainLosses, validationLosses, trainAccuracies, validationAccuracies):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        epochsRange = range(1, len(trainLosses) + 1)
        
        ax1.plot(epochsRange, trainLosses, 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochsRange, validationLosses, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(epochsRange, trainAccuracies, 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochsRange, validationAccuracies, 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.outputDir, 'TrainingHistory.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def ConfusionMatrixPlot(self, actuals, predictions):
        cm = confusion_matrix(actuals, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Stress', 'Stress'], 
                    yticklabels=['No Stress', 'Stress'],
                    cbar_kws={'label': 'Count'},
                    square=True, linewidths=0.5)
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        
        total = cm.sum()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                percentage = cm[i, j] / total * 100
                plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                        ha='center', va='center', fontsize=10, color='gray')
        
        plt.savefig(os.path.join(self.outputDir, 'ConfusionMatrix.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def MetricsComparisonPlot(self, metrics):
        metricsNames = ['Accuracy', 'F1 Score', 'Recall', 'Precision']
        values = [metrics['accuracy'], metrics['f1'], metrics['recall'], metrics['precision']]
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(metricsNames, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', 
                    fontweight='bold', fontsize=12)
        
        plt.title('Model Performance Metrics', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Score', fontsize=12)
        plt.ylim(0, 1.1)
        plt.grid(axis='y', alpha=0.3)
        
        plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, 
                label='Random Baseline (0.5)', linewidth=2)
        plt.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.outputDir, 'MetricsComparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
