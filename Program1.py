import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report
from LSTM import LSTM



maxWordsPhrase = x.apply(lambda x: len(x.split(' '))).max()


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
print("Accuracy:", accuracy_score(actuals, predictions))
print("Training completed.")