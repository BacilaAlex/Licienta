import torch
from sklearn.metrics import f1_score, recall_score, precision_score
from PlotGenerator import PlotGenerator
from ModelSaver import ModelSaver

class Trainer:
    def __init__(self, device, model, criterion, optimizer, scheduler, metricMonitor,trainData, testData):
        self.device = device
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metricMonitor = metricMonitor
        self.trainData = trainData
        self.testData = testData
        self.trainLosses = []
        self.validationLosses = []
        self.trainAccuracies = []
        self.validationAccuracies = []
        self.plotGenerator = PlotGenerator(device, model, testData)
        self.modelSaver = ModelSaver(model, optimizer, scheduler)

    def Train(self, epochs):
        print("Starting training...")
        print(f"Training for {epochs} epochs")
        print("-" * 50)
        
        for epoch in range(epochs):
            trainMetrics = self.TrainEpoch()
            validationMetrics = self.ValidateEpoch()  
            
            self.trainLosses.append(trainMetrics['loss'])
            self.validationLosses.append(validationMetrics['loss'])
            self.trainAccuracies.append(trainMetrics['accuracy'])
            self.validationAccuracies.append(validationMetrics['accuracy'])

            if epoch % 5 == 0 :
                self.PrintEpochResults(epoch, epochs, trainMetrics, validationMetrics)
            
            metrics = {
                'accuracy': validationMetrics['accuracy'] / 100,
                'f1': validationMetrics['f1'],                
                'recall': validationMetrics['recall'],
                'precision': validationMetrics['precision'],
                'val_loss': validationMetrics['loss'],
                'train_loss': trainMetrics['loss']
            }

            self.modelSaver.SaveModel(
                epoch=epoch,
                metrics=metrics,
                trainingLosses=self.trainLosses,
                validationLosses=self.validationLosses,
                trainingAccuracies=self.trainAccuracies,
                validationAccuracies=self.validationAccuracies
            )

            self.scheduler.step(validationMetrics.get(self.metricMonitor))
            print(f"Learning rate adjusted to: {self.scheduler.get_last_lr()[0]:.6f}")
        
        print("\nTraining completed!")

    def TrainEpoch(self):
        self.model.train()
        epochLoss = 0.0
        epochSamples = 0
        correctTrain = 0
        totalTrain = 0
        
        for data, target in self.trainData:
            data, target = data.to(self.device), target.to(self.device)
            target = target.view(-1, 1)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            epochLoss += loss.item() * len(data)
            epochSamples += len(data)
            
            predicted = (torch.sigmoid(output) > 0.5).float()
            totalTrain += target.size(0)
            correctTrain += (predicted == target).sum().item()
        
        averageLoss = epochLoss / epochSamples
        accuracy = 100 * correctTrain / totalTrain
        
        return {'loss': averageLoss, 'accuracy': accuracy}

    def ValidateEpoch(self):
        self.model.eval()
        totalValidationLoss = 0.0
        totalValidationSamples = 0
        correctValidation = 0
        totalValidation = 0
        
        predictions, actuals = [], []
        
        with torch.no_grad():
            for data, target in self.testData:
                data, target = data.to(self.device), target.to(self.device)
                target = target.view(-1, 1)
                
                output = self.model(data)
                validationLoss = self.criterion(output, target)
                totalValidationLoss += validationLoss.item() * len(data)
                totalValidationSamples += len(data)
                
                predicted = (torch.sigmoid(output) > 0.5).float()
                totalValidation += target.size(0)
                correctValidation += (predicted == target).sum().item()
                
                predictions.extend(predicted.squeeze().cpu().numpy().tolist())
                actuals.extend(target.squeeze().cpu().numpy().tolist())
        
        averageLoss = totalValidationLoss / totalValidationSamples
        accuracy = 100 * correctValidation / totalValidation
        
        predictions = [int(p) for p in predictions]
        actuals = [int(a) for a in actuals]

        f1 = f1_score(actuals, predictions, zero_division=0)
        recall = recall_score(actuals, predictions, zero_division=0)
        precision = precision_score(actuals, predictions, zero_division=0)
    
        return {'loss': averageLoss, 'accuracy': accuracy, 'f1': f1,
                'recall': recall, 'precision': precision,
                'predictions': predictions, 'actuals': actuals}
    
    def PrintEpochResults(self, epoch, totalEpochs, trainMetrics, valMetrics):
        print(f"Epoch {epoch:3d}/{totalEpochs} | "
            f"Train Loss: {trainMetrics['loss']:.4f} | "
            f"Val Loss: {valMetrics['loss']:.4f} | "
            f"Train Acc: {trainMetrics['accuracy']:.2f}% | "
            f"Val Acc: {valMetrics['accuracy']:.2f}% | "
            f"F1: {valMetrics['f1']:.4f} | "
            f"Recall: {valMetrics['recall']:.4f} | "
            f"Precision: {valMetrics['precision']:.4f}")
    
    def Evaluate(self):
        self.modelSaver.LoadModel(self.device)
        self.model.to(self.device)
        self.model.eval() 

        self.plotGenerator.Generate(
            self.trainLosses, 
            self.validationLosses, 
            self.trainAccuracies,            
            self.validationAccuracies
        )

        metrics = self.modelSaver.metrics
        epoch = self.modelSaver.bestEpoch

        print("\n" + "-"*70)
        print("Best Model Summary")
        print("-"*70)
        print(f"Epoch:        {epoch}")
        print(f"Accuracy:     {metrics.get('accuracy', 'N/A'):.4f}")
        print(f"F1 Score:     {metrics.get('f1', 'N/A'):.4f}")
        print(f"Recall:       {metrics.get('recall', 'N/A'):.4f}")
        print(f"Precision:    {metrics.get('precision', 'N/A'):.4f}")
        print(f"Validation Loss:   {metrics.get('val_loss', 'N/A'):.4f}")
        print(f"Training Loss:     {metrics.get('train_loss', 'N/A'):.4f}")
        print("-"*70)
