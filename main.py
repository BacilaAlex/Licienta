import torch
import pandas as pd
from TextCleaner import TextCleaner
from Trainer import Trainer
import torch.nn as nn
from TextProcessor import TextProcessor
from LSTM import LSTM
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

def main():
    textCleaner = TextCleaner()
    textProcessor = TextProcessor()

    epochs = 150
    batchSize = 128
    embeddingSize = 325
    hiddenSize = 128
    layers = 6
    dropout = 7e-1
    learningRate = 1e-3
    
    df = GetArticleData()
    x = df["text"].apply(lambda x: textCleaner.GetCleanedData(x))
    y = df["label"]

    vocabulary = textProcessor.GenerateVocabulary(x)

    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3, random_state=42)

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
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

    trainer = Trainer(device, model, criterion, optimizer, trainData, testData)
    trainer.Train(epochs)
    trainer.Evaluate()

    pass

def GetArticleData():
    df1 = pd.read_csv(r"dreaddit/dreaddit-train.csv")
    df3 = pd.read_csv(r"dreaddit/dreaddit-test.csv")
    df = pd.concat([df1, df3], ignore_index=True)
    return df

if __name__ == "__main__":
    main()