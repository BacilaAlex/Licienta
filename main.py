import torch
import pandas as pd
from TextCleaner import TextCleaner
from Trainer import Trainer
import torch.nn as nn
from TextProcessor import TextProcessor
from LSTM import LSTM
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

def GetArticleData():
    df1 = pd.read_csv(r"dreaddit/dreaddit-train.csv")
    df3 = pd.read_csv(r"dreaddit/dreaddit-test.csv")
    df = pd.concat([df1, df3], ignore_index=True)
    return df


def main():
    textCleaner = TextCleaner()
    textProcessor = TextProcessor()

    df = GetArticleData()
    
    x = df["text"].apply(lambda x: textCleaner.GetCleanedData(x))
    y = df["label"]

    vocabulary = textProcessor.GenerateVocabulary(x)

    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=42)

    xTrain  = textProcessor.GeneratePaddings(xTrain ,vocabulary)
    xTest = textProcessor.GeneratePaddings(xTest, vocabulary)

    yTrain = torch.tensor(yTrain.values, dtype=torch.float)
    yTest = torch.tensor(yTest.values, dtype=torch.float)
    
    batchSize = 32
    trainData = list(zip(xTrain, yTrain))
    testData = list(zip(xTest, yTest))
    trainData = DataLoader(trainData, batch_size=batchSize)
    testData = DataLoader(testData, batch_size=batchSize)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    outputSize = 1
    hiddenSize = 128
    layers = 2
    epochs = 15

    model = LSTM(len(vocabulary), outputSize, layers, hiddenSize).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    trainer = Trainer(device, model, criterion, optimizer, layers, hiddenSize, trainData, testData)
    trainer.train(epochs)
    trainer.evaluate()

    pass

if __name__ == "__main__":
    main()