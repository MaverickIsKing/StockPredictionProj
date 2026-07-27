from model import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler
import features
import os


scaler = StandardScaler()

directory = os.getcwd()
file_name = "NVDAGOOGTSLAMULITE2026-07-14-19-55-00.csv"
df = pd.read_csv(os.path.join(directory,file_name))
df_lite = df.loc[df['Ticker']=='LITE']
df_lite.reset_index(drop=True, inplace=True)
df_lite.set_index('Date', inplace=True)
df_lite.ffill(inplace=True)

def prepareDataForLSTM(df, label, memLength):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    data = []
    meanstd = []
    for i in range(len(df)-memLength):
        
        sequence = df[label][i:i+memLength].values.reshape(-1, len(label))
        scaler.fit(sequence)
        scaledSeq = scaler.transform(sequence)
        data.append(scaledSeq)
        meanstd.append([scaler.mean_,np.sqrt(scaler.var_)])
    return np.array(data), np.array(meanstd)

data, meanstd = prepareDataForLSTM(df_lite, ['Open','High','Low','Adj Close','Volume'], 30)

train_size = int(0.8*len(data))

X_train = torch.from_numpy(data[:train_size, :-1, :]).type(torch.Tensor).to(device)
X_meanstd_train = torch.from_numpy(meanstd[:train_size, :-1, :]).type(torch.Tensor)
Y_train = torch.from_numpy(data[:train_size, -1, :]).type(torch.Tensor).to(device)
Y_meanstd_train = torch.from_numpy(meanstd[:train_size, -1, :]).type(torch.Tensor)
X_test = torch.from_numpy(data[train_size:, :-1, :]).type(torch.Tensor).to(device)
X_meanstd_test = torch.from_numpy(meanstd[train_size:, :-1, :]).type(torch.Tensor)
Y_test = torch.from_numpy(data[train_size:, -1, :]).type(torch.Tensor).to(device)
Y_meanstd_test = torch.from_numpy(meanstd[train_size:, -1, :]).type(torch.Tensor)
model = LSTMModel(input_dim=5, hidden_dim=128, num_layers=4, output_dim=5, dropout=0.2).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100

for i in range(num_epochs):
    Y_train_pred  = model(X_train)

    loss = criterion(Y_train_pred, Y_train)

    if i % 25 == 0:
        print(i, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model.eval()

torch.save(model.state_dict(), os.path.join(directory,'MyLSTMforStockPred.pth'))

model2 = LSTMModel(input_dim=5, hidden_dim=128, num_layers=4, output_dim=5, dropout=0.2)

model2.load_state_dict(torch.load(os.path.join(directory,'MyLSTMforStockPred.pth')))
model2.eval()

Y_train_pred = model2(X_train)
Y_test_pred = model2(X_test)

Y_train_pred = Y_train_pred.detach().cpu().numpy()
Y_train = Y_train.detach().cpu().numpy()
Y_test_pred = Y_test_pred.detach().cpu().numpy()
Y_test = Y_test.detach().cpu().numpy()
Y_meanstd_train = Y_meanstd_train.detach().cpu().numpy()
Y_meanstd_test = Y_meanstd_test.detach().cpu().numpy()




for i in range(len(Y_train)):
    Y_train[i] = Y_train[i]*Y_meanstd_train[i][1]+Y_meanstd_train[i][0]
    Y_train_pred[i] = Y_train_pred[i]*Y_meanstd_train[i][1]+Y_meanstd_train[i][0]

for i in range(len(Y_test)):
    Y_test[i] = Y_test[i]*Y_meanstd_test[i][1]+Y_meanstd_test[i][0]
    Y_test_pred[i] = Y_test_pred[i]*Y_meanstd_test[i][1]+Y_meanstd_test[i][0]

train_rmse = root_mean_squared_error(Y_train[:, 3], Y_train_pred[:, 3])
test_rmse = root_mean_squared_error(Y_test[:, 3], Y_test_pred[:, 3])

fig = plt.figure(figsize=(12,10))

gs = fig.add_gridspec(4,1)

ax1 = fig.add_subplot(gs[:3, 0])
ax1.plot(df.iloc[int(-len(Y_test)*0.1):].index, Y_test[int(-len(Y_test)*0.1):,3], color = 'blue', label='Actual Price')
ax1.plot(df.iloc[int(-len(Y_test)*0.1):].index, Y_test_pred[int(-len(Y_test)*0.1):,3], color = 'green', label='Predicted Price')
ax1.legend()
plt.title(f"Stock Price Prediction")
plt.xlabel('Date')
plt.ylabel('Price')

ax2 = fig.add_subplot(gs[3, 0])
ax2.axhline(test_rmse, color='blue', linestyle='--', label='RMSE')
ax2.plot(df[int(-len(Y_test)*0.1):].index, abs(Y_test[int(-len(Y_test)*0.1):,3] - Y_test_pred[int(-len(Y_test)*0.1):,3]), 'r', label='Prediction Error')
ax2.legend()
plt.title('Prediction Error')
plt.xlabel('Date')
plt.ylabel('Error')
plt.tight_layout()

plt.show()