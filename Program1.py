import nltk
import re
import pandas as pd
import string
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords

from LSTM import LSTM

df1 = pd.read_csv(r"dreaddit/dreaddit-train.csv")
df3 = pd.read_csv(r"dreaddit/dreaddit-test.csv")
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

x = df["text"].apply(clean)
y = df["label"]

maxWordsPhrase = x.apply(lambda x: len(x.split(' '))).max()

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=42)


tokenizer = get_tokenizer('basic_english')

def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(xTest), min_freq = 3,specials=["<pad>","<unk>"])

vocab.set_default_index(vocab["<unk>"])

vocab.get_itos()

def process_text(text, vocab, tokenizer):
    tokens = tokenizer(text)
    return torch.tensor([vocab[token] for token in tokens], dtype=torch.long)

# Transform the text data
xTrain_tokens = [process_text(text, vocab, tokenizer) for text in xTrain]
xTest_tokens = [process_text(text, vocab, tokenizer) for text in xTest]

# Pad sequences
xTrain_padded = pad_sequence(xTrain_tokens, batch_first=True, padding_value=vocab['<pad>'])
xTest_padded = pad_sequence(xTest_tokens, batch_first=True, padding_value=vocab['<pad>'])

# Convert labels to tensors
yTrain_tensor = torch.tensor(yTrain.values, dtype=torch.float32)
yTest_tensor = torch.tensor(yTest.values, dtype=torch.float32)

# Create data loader
batch_size = 32
train_data = list(zip(xTrain_padded, yTrain_tensor))
test_data = list(zip(xTest_padded, yTest_tensor))
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab_size = len(vocab)
output_size = 1  # Binary classification
hidden_size = 128
num_layers = 2

model = LSTM(vocab_size, output_size, num_layers, hidden_size).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
print("Starting training...")

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Initialize hidden state and cell state
        hidden = torch.zeros(num_layers, data.size(0), hidden_size).to(device)
        cell = torch.zeros(num_layers, data.size(0), hidden_size).to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output, hidden, cell = model(data, hidden, cell)
        output = output[:, -1, :]  # Take the last output
        loss = criterion(output.squeeze(), target)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch: {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')

# Evaluation
model.eval()
predictions = []
actuals = []

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        hidden = torch.zeros(num_layers, data.size(0), hidden_size).to(device)
        cell = torch.zeros(num_layers, data.size(0), hidden_size).to(device)
        
        output, _, _ = model(data, hidden, cell)
        output = output[:, -1, :]  # Take the last output
        pred = torch.sigmoid(output.squeeze()) > 0.5
        
        predictions.extend(pred.cpu().numpy())
        actuals.extend(target.cpu().numpy())

print("\nTest Results:")
print(classification_report(actuals, predictions))
