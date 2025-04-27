import torch
from sklearn.metrics import accuracy_score, classification_report

class Trainer:
    def __init__(self,device, model, criterion, optimizer , trainData, testData, embedding_size):
        self.device = device
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.trainData = trainData
        self.testData = testData
        self.embedding_size = embedding_size

    def Train(self, epochs):
        print("[INFO] Training started...")
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            total_samples = 0
            
            for data, target in self.trainData:
                data, target = data.to(self.device), target.to(self.device)
                # print("Data shape:", data.shape, "Target shape:", target.shape)
                self.optimizer.zero_grad()

                # Get output and hidden state from model
                output = self.model(data)
                output = output.squeeze()  # Ensure output matches target dimensions
                
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item() * len(data)
                total_samples += len(data)

            # Validation phase
            self.model.eval()
            total_val_loss = 0.0
            total_val_samples = 0
            
            with torch.no_grad():
                for data, target in self.testData:
                    data, target = data.to(self.device), target.to(self.device)
                    # print(data.shape, target.shape)

                    output = self.model(data)
                    output = output.squeeze()  # Ensure output matches target dimensions
                    
                    val_loss = self.criterion(output, target)
                    total_val_loss += val_loss.item() * len(data)
                    total_val_samples += len(data)

            avg_train_loss = total_loss / total_samples
            avg_val_loss = total_val_loss / total_val_samples
            
            print(f'Epoch: {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        print("[INFO] Training complete.")

    def Evaluate(self):
        self.model.eval()
        predictions, actuals = [], []
        with torch.no_grad():
            for data, target in self.testData:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                output = output.squeeze()  # Ensure output matches target dimensions
                prediction = (torch.sigmoid(output) > 0.5).float()

                predictions.extend(prediction.cpu().numpy().tolist())
                actuals.extend(target.cpu().numpy().tolist())
        
        print("\nTest Results:")
        predictions = [int(p) for p in predictions]
        actuals = [int(a) for a in actuals]
        print(classification_report(actuals, predictions))
        print("Accuracy:", accuracy_score(actuals, predictions))
