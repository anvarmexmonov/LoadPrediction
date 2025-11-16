import torch
import pandas as pd
from model import BinaryClassifier
from testing_validation_loader import test_loader, test_ids




print("Loading trained model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BinaryClassifier()
model.load_state_dict(torch.load('final_model1.pth', map_location=device))
model.to(device)
model.eval()

# Generate predictions
print("Generating predictions...\n")
all_probs = []

with torch.no_grad():
    for X in test_loader:
        X = X.to(device)
        outputs = model(X)
        all_probs.extend(outputs.cpu().numpy())

# Create predictions (0 or 1)
all_preds = [1 if prob > 0.55 else 0 for prob in all_probs]

# Create results dataframe
results_df = pd.DataFrame({
    'id': test_ids,
    'probability': all_probs,
    'prediction': all_preds
})

results_df.to_csv('result.csv', index=False)

print("=" * 50)
print("PREDICTIONS GENERATED")
print("=" * 50)
print(f"Total predictions: {len(all_preds)}")
print(f"Predicted class 0: {all_preds.count(0)}")
print(f"Predicted class 1: {all_preds.count(1)}")
print(f"\nResults saved to: predictions.csv")
print("=" * 50)

print("\nFirst 10 predictions:")
print(results_df.head(10).to_string(index=False))
results_df.to_csv('predictions.csv', index=False)
