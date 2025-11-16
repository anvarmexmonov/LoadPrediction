from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd

df = pd.read_csv('final_data_no_ref.csv')
X = df.drop('default', axis=1)
y = df['default']

class BinaryDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


scaler = StandardScaler()
X_train = scaler.fit_transform(X)


train_loader = DataLoader(BinaryDataset(X_train, y.values), batch_size=32, shuffle=True)
