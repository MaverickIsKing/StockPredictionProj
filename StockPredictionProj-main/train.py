from model import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler
import features


scaler = StandardScaler()

df = pd.read_csv('NVDAGOOGTSLAMULITE2026-07-14-19-55-00.csv')
df_lite = df.loc[df['Ticker']=='LITE']
df_lite.reset_index(drop=True, inplace=True)
df_lite.set_index('Date', inplace=True)
df_lite.ffill(inplace=True)

def prepareDataForLSTM(df, label, memLength):
    # scaler.fit(df[label].values.reshape(-1, 1))
    data = []
    for i in range(len(df)-memLength):
        sequence = df[label][i:i+memLength].values.reshape(-1, 1)
        # scaledSeq = scaler.transform(sequence)
        data.append(sequence)
    return np.array(data)

data = prepareDataForLSTM(df_lite, "Adj Close", 100)

train_size = int(0.8*len(data))

X_train = torch.from_numpy(data[:train_size, :-1, :]).type(torch.Tensor).to(device)
Y_train = torch.from_numpy(data[:train_size, -1, :]).type(torch.Tensor).to(device)
X_test = torch.from_numpy(data[train_size:, :-1, :]).type(torch.Tensor).to(device)
Y_test = torch.from_numpy(data[train_size:, -1, :]).type(torch.Tensor).to(device)

model = LSTMModel(input_dim=1, hidden_dim=32, num_layers=2, output_dim=1, dropout=0.1).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 1000

for i in range(num_epochs):
    Y_train_pred  = model(X_train)

    loss = criterion(Y_train_pred, Y_train)

    if i % 25 == 0:
        print(i, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model.eval()

Y_train_pred = model(X_train)
Y_test_pred = model(X_test)

Y_train_pred = Y_train_pred.detach().cpu().numpy()
Y_train = Y_train.detach().cpu().numpy()
Y_test_pred = Y_test_pred.detach().cpu().numpy()
Y_test = Y_test.detach().cpu().numpy()

# Y_train_pred = scaler.inverse_transform(Y_train_pred)
# Y_test_pred = scaler.inverse_transform(Y_test_pred)
# Y_test = scaler.inverse_transform(Y_test)

train_rmse = root_mean_squared_error(Y_train[:, 0], Y_train_pred[:, 0])
test_rmse = root_mean_squared_error(Y_test[:, 0], Y_test_pred[:, 0])

fig = plt.figure(figsize=(12,10))

gs = fig.add_gridspec(4,1)

ax1 = fig.add_subplot(gs[:3, 0])
ax1.plot(df.iloc[-len(Y_test):].index, Y_test, color = 'blue', label='Actual Price')
ax1.plot(df.iloc[-len(Y_test):].index, Y_test_pred, color = 'green', label='Predicted Price')
ax1.legend()
plt.title(f"Stock Price Prediction")
plt.xlabel('Date')
plt.ylabel('Price')

ax2 = fig.add_subplot(gs[3, 0])
ax2.axhline(test_rmse, color='blue', linestyle='--', label='RMSE')
ax2.plot(df[-len(Y_test):].index, abs(Y_test - Y_test_pred), 'r', label='Prediction Error')
ax2.legend()
plt.title('Prediction Error')
plt.xlabel('Date')
plt.ylabel('Error')
plt.tight_layout()

plt.show()