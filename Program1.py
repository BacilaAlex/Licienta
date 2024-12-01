import nltk
import re
import pandas as pd
import string

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
from nltk.corpus import stopwords

df1 = pd.read_csv(r"D:\Scoala\Anul 4\Licienta\dreaddit\dreaddit-train.csv")
df3 = pd.read_csv(r"D:\Scoala\Anul 4\Licienta\dreaddit\dreaddit-test.csv")
df = pd.concat([df1, df3], ignore_index=True)

stemmer = nltk.SnowballStemmer("english")
stopwords = set(stopwords.words("english"))


def clean(text):
    # Convert the input to a string and make all text lowercase.
    text = str(text).lower()
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
    # Remove words that contain numbers (e.g., "123abc", "abc123").
    text = re.sub(r'\w*\d\w*', '', text)
    # Split the text into words and remove common stopwords from the text.
    text = [word for word in text.split(' ') if word not in stopwords]
    # Rejoin the list of words into a single string with spaces between them.
    text = " ".join(text)
    # Split the text into words again and apply stemming to reduce words to their root form.
    text = [stemmer.stem(word) for word in text.split(' ')]
    # Rejoin the list of stemmed words into a single string with spaces between them.
    text = " ".join(text)
    # Return the fully cleaned and processed text.
    return text


df["label"] = df["label"].map({0: "No Stress", 1: "Stress"})
df["text"] = df["text"].apply(clean)

x = df["text"]
y = df["label"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

vect = TfidfVectorizer(max_features=10000, stop_words='english')
x_train_vectorized = vect.fit_transform(x_train)
x_test_vectorized = vect.transform(x_test)

model1 = DecisionTreeClassifier()
model2 = LogisticRegression()
model3 = SVC(kernel='linear', random_state=42)

model1.fit(x_train_vectorized, y_train)
y_pred = model1.predict(x_test_vectorized)
accuracy=accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report Model1:")
print(classification_report(y_test, y_pred))

model2.fit(x_train_vectorized, y_train)
y_pred = model2.predict(x_test_vectorized)
accuracy=accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report Model2:")
print(classification_report(y_test, y_pred))

model3.fit(x_train_vectorized, y_train)
y_pred = model3.predict(x_test_vectorized)
accuracy=accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report Model3:")
print(classification_report(y_test, y_pred))

user = "Sometime I feel like I need some help"
df2 = vect.transform([user]).toarray()
output = model3.predict(df2)
print(output)
