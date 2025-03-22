import pandas as pd
from TextCleaner import TextCleaner
from TextProcessor import TextProcessor
from sklearn.model_selection import train_test_split

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

    print(x)
    print(y)

    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=42)

    textProcessor.GeneratePaddings(x)
    pass

if __name__ == "__main__":
    main()