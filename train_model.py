import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error
# the nomrmalized data

df = pd.read_csv("SPUS_normalized.csv")
df['date'] = pd.to_datetime(df['date'])


# spliting
#    train: 2023–2024
#    test: 2025

train_df = df[df['date'].dt.year.isin([2023, 2024])]
test_df = df[df['date'].dt.year == 2025]

# features (X) and target (y)

features = ['open', 'high', 'low', 'volume']
target = 'close'

X_train = train_df[features].values
y_train = train_df[target].values

X_test = test_df[features].values
y_test = test_df[target].values

# convert to tensors

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# neural network

class PricePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)

model = PricePredictor()

# training setup

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# training on 2023–2024
epochs = 50
for epoch in range(epochs):
    optimizer.zero_grad()
    preds = model(X_train)
    loss = criterion(preds, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.6f}")

# testing on 2025

model.eval()
with torch.no_grad():
    predictions_2025 = model(X_test)

mae = mean_absolute_error(
    y_test.numpy(),
    predictions_2025.numpy()
)

print("2025 MAE (normalized):", mae)
