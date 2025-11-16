from full_data_loader import full_training_loader
import torch
from model import BinaryClassifier
import torch.optim as optim
import torch.nn as nn

model = BinaryClassifier()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
criterion = nn.BCELoss()
epochs = 50

print("Training on full dataset...\n")

for epoch in range(epochs):
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for X, y in full_training_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = (outputs > 0.5).float()
        train_correct += (preds == y).sum().item()
        train_total += y.size(0)

    train_loss /= len(full_training_loader)
    train_acc = train_correct / train_total

    print(f'Epoch [{epoch + 1}/{epochs}] | Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f}')

# Save final model
torch.save(model.state_dict(), 'final_model1.pth')
print(f'\nTraining completed! Final Accuracy: {train_acc:.4f}')
print('Model saved as final_model.pth')