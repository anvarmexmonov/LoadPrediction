from sklearn.model_selection import train_test_split
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


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

train_loader = DataLoader(BinaryDataset(X_train, y_train.values), batch_size=32, shuffle=True)
val_loader = DataLoader(BinaryDataset(X_test, y_test.values), batch_size=32)