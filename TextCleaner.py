import re
import string
import nltk
import contractions
from nltk.corpus import stopwords

class TextCleaner:
    def __init__(self):
        nltk.download('stopwords')
        self.stemmer = nltk.SnowballStemmer("english")
        self.stopwords = set(stopwords.words("english"))

    def GetCleanedData(self, text):
        text = str(text).lower()
        text = contractions.fix(text)
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>+', '', text)
        text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub(r'\n', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\w*\d\w*', '', text)
        text = " ".join([word for word in text.split() if word not in self.stopwords])
        text = " ".join([self.stemmer.stem(word) for word in text.split()])
        return text
