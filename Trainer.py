import torch
from sklearn.metrics import accuracy_score, classification_report

class Trainer:
    def __init__(self,device, model, criterion, optimizer , layers, hiddenSize, trainData, testData):
        self.device = device
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.layers = layers
        self.hiddenSize = hiddenSize
        self.trainData = trainData
        self.testData = testData

    def train(self, epochs):
        print("Starting training...")
        for epoch in range(epochs):
            self.model.train()
            totalLoss = 0
            for data, target in self.trainData:
                data, target = data.to(self.device), target.to(self.device)
                
                hidden = torch.zeros(self.layers, data.size(0), self.hiddenSize).to(self.device)
                cell = torch.zeros(self.layers, data.size(0), self.hiddenSize).to(self.device)
                
                self.optimizer.zero_grad()

                output, hidden, cell = self.model(data, hidden, cell)
                output = output[:, -1, :] 
                output = output.squeeze() 

                loss = self.criterion(output, target)
                loss.backward()

                self.optimizer.step()
                
                totalLoss += loss.item()
            
            avg_loss = totalLoss / len(self.trainData)
            print(f'Epoch: {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}')

    def evaluate(self):
        self.model.eval()
        predictions, actuals = [], []
        with torch.no_grad():
            for data, target in self.testData:
                data, target = data.to(self.device), target.to(self.device)

                hidden = torch.zeros(self.layers, data.size(0), self.hiddenSize).to(self.device)
                cell = torch.zeros(self.layers, data.size(0), self.hiddenSize).to(self.device)
                
                output, _, _ = self.model(data, hidden, cell)
                output = output[:, -1, :]
                output = output.squeeze()

                prediction = (torch.sigmoid(output) > 0.5).float()

                predictions.extend(prediction.cpu().numpy().tolist())
                actuals.extend(target.cpu().numpy().tolist())
        
        print("\nTest Results:")
        predictions = [int(p) for p in predictions]
        actuals = [int(a) for a in actuals]
        print(classification_report(actuals, predictions))
        print("Accuracy:", accuracy_score(actuals, predictions))
