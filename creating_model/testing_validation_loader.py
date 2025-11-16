import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


class TestDataset(Dataset):
    def __init__(self, X):
        self.X = torch.FloatTensor(X)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]

print("Loading test data...")
test_df = pd.read_csv('final_validation_set.csv')

if 'id' in test_df.columns:
    test_ids = test_df['id']
    X_test = test_df.drop('id', axis=1)
else:
    test_ids = range(len(test_df))
    X_test = test_df

scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)

test_loader = DataLoader(TestDataset(X_test_scaled), batch_size=32, shuffle=False)