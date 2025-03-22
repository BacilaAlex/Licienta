import pandas as pd
from TextCleaner import TextCleaner
from TextProcessor import TextProcessor


def GetArticleData():
    df1 = pd.read_csv(r"dreaddit/dreaddit-train.csv")
    df3 = pd.read_csv(r"dreaddit/dreaddit-test.csv")
    df = pd.concat([df1, df3], ignore_index=True)
    return df


def main():
    textCleaner = TextCleaner()
    textProcessor = TextProcessor()

    df = GetArticleData()
    
    x = textCleaner.GetCleanedData(df["text"])
    y = df["label"]

    text_data = [
    "Hello world!",
    "This is a test.",
    "Cava is great.",
    "I love coding.",
    "Natural language processing is fun.",
    "Check out this link: https://example.com",
    "Text preprocessing is important.",
    "Python is awesome.",
    "How do you do?",
    "Let's test this out!"
    ]

    textProcessor.GeneratePaddings(text_data)

    pass

if __name__ == "__main__":
    main()