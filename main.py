import torch
import pandas as pd
from TextCleaner import TextCleaner
from Trainer import Trainer
import torch.nn as nn
from TextProcessor import TextProcessor
from LSTM import LSTM
from xLSTM.xLSTMClassifier import xLSTMClassifier
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

def main():
    textCleaner = TextCleaner()
    textProcessor = TextProcessor()

    batchSize = 64
    epochs = 20
    learningRate = 0.01
    embeddingSize = 64
    layers = ['s', 'm', 'm',]
    
    df = GetArticleData()
    x = df["text"].apply(lambda x: textCleaner.GetCleanedData(x))
    y = df["label"]

    vocabulary = textProcessor.GenerateVocabulary(x)

    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=42)

    xTrain  = textProcessor.GeneratePaddings(xTrain ,vocabulary)
    xTest = textProcessor.GeneratePaddings(xTest, vocabulary)

    yTrain = torch.tensor(yTrain.values, dtype=torch.float)
    yTest = torch.tensor(yTest.values, dtype=torch.float)
    
    trainData = list(zip(xTrain, yTrain))
    testData = list(zip(xTest, yTest))
    trainData = DataLoader(trainData, batch_size=batchSize, shuffle=True)
    testData = DataLoader(testData, batch_size=batchSize)

    print("Train data shape:", len(trainData.dataset), "Test data shape:", len(testData.dataset))
    print ("Train data shape:", len(trainData), "Test data shape:", len(testData))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model = LSTM(len(vocabulary), embeddingSize, hiddenSize, outputSize, layers).to(device)
    print("[INFO] Initializing model...")
    model = xLSTMClassifier(layers,  batchSize, len(vocabulary),embeddingSize, depth=4, factor=2, dropout_head=0.1).to(device)
    print("[INFO] Model initialized.")
    criterion = nn.BCEWithLogitsLoss()  # Changed to BCEWithLogitsLoss
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

    trainer = Trainer(device, model, criterion, optimizer, trainData, testData, embeddingSize)
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