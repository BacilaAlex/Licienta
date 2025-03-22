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
        # Convert the input to a string and make all text lowercase.
        text = str(text).lower()
        # Expand contractions (e.g., "don't" -> "do not")
        text = contractions.fix(text)
        # Remove text enclosed in square brackets, including the brackets themselves.
        text = re.sub(r'\[.*?\]', '', text)
        # Remove URLs that start with "http://" or "https://" or begin with "www."
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Remove HTML tags (e.g., <div>, <p>, etc.).
        text = re.sub(r'<.*?>+', '', text)
        # Remove punctuation characters (e.g., ., !, ?, etc.).
        text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
        # Remove newline characters (\n) from the text.
        text = re.sub(r'\n', '', text)
        # Remove any remaining non-word characters except spaces.
        text = re.sub(r'[^\w\s]', '', text)
        # Remove words that contain numbers (e.g., "123abc", "abc123").
        text = re.sub(r'\w*\d\w*', '', text)
        # Rejoin and Split the text into words and remove common stopwords from the text.
        text = " ".join([word for word in text.split() if word not in self.stopwords])
        # Rejoin and Split the text into words again and apply stemming to reduce words to their root form.
        text = " ".join([self.stemmer.stem(word) for word in text.split()])
        # Return the fully cleaned and processed text.
        return text
