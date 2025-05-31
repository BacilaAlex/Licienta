import json
import torch
import pandas as pd
from TextCleaner import TextCleaner
from Trainer import Trainer
import torch.nn as nn
from TextProcessor import TextProcessor
from LSTM import LSTM
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

def main():
    textCleaner = TextCleaner()
    textProcessor = TextProcessor()
    config = LoadConfig()
    
    epochs = config["Training"]["Epochs"]
    batchSize = config["Training"]["BatchSize"]
    embeddingSize = config["Model"]["EmbeddingSize"]
    hiddenSize = config["Model"]["HiddenSize"]
    layers = config["Model"]["Layers"]
    dropout = config["Model"]["Dropout"]
    learningRate = config["Optimizer"]["LearningRate"]
    weightDecay = config["Optimizer"]["WeightDecay"]
    mode = config["Scheduler"]["Mode"]
    factor = config["Scheduler"]["Factor"]
    patience = config["Scheduler"]["Patience"]
    minimumLearningRate = config["Scheduler"]["MinimumLearningRate"]
    metricMonitor = config["Scheduler"]["MetricMonitor"]

    df = GetArticleData()
    x = df["text"].apply(lambda x: textCleaner.GetCleanedData(x))
    y = df["label"]

    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=42)

    vocabulary = textProcessor.GenerateVocabulary(x)

    xTrain  = textProcessor.GeneratePaddings(xTrain ,vocabulary)
    xTest = textProcessor.GeneratePaddings(xTest, vocabulary)

    yTrain = torch.tensor(yTrain.values, dtype=torch.float)
    yTest = torch.tensor(yTest.values, dtype=torch.float)
    
    trainData = list(zip(xTrain, yTrain))
    testData = list(zip(xTest, yTest))

    trainData = DataLoader(trainData, batch_size=batchSize, shuffle=True)
    testData = DataLoader(testData, batch_size=batchSize)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LSTM(len(vocabulary), embeddingSize, hiddenSize, layers, dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate, weight_decay=weightDecay)
    scheduler = ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience, min_lr= minimumLearningRate)

    trainer = Trainer(device, model, criterion, optimizer,scheduler, metricMonitor, trainData, testData)
    # trainer = Trainer(device, model, criterion, optimizer, trainData, testData)
    trainer.Train(epochs)
    trainer.Evaluate()


def GetArticleData():
    df1 = pd.read_csv(r"Dreaddit/dreaddit-train.csv")
    df3 = pd.read_csv(r"Dreaddit/dreaddit-test.csv")
    df = pd.concat([df1, df3], ignore_index=True)
    return df

def LoadConfig(configPath="config.json"):
    with open(configPath, 'r') as f:
        config = json.load(f)
    return config


if __name__ == "__main__":
    main()