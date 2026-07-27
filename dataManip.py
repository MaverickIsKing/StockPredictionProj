import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error
import features
import os

directory = os.getcwd()
file_name = "NVDAGOOGTSLAMULITE2026-07-14-19-55-00.csv"
df = pd.read_csv(os.path.join(directory,file_name))
df_lite = df.loc[df['Ticker']=='LITE']
df_lite.reset_index(drop=True, inplace=True)

# df_log = features.log_change(df_lite,5)
# df_lite_ma5 = features.calculate_ma(df_lite,5)
# df_lite_pct_change60 = features.pct_change(df_lite,5)
# df_lite_ema30 = features.calculate_ema(df_lite, 30)
# pd.set_option('display.max_rows', None)
# print(df_lite_ema30)
# df_lite.set_index('Date', inplace=True)
# print(df_lite)
# plt.plot(df_lite['Adj Close'])
# plt.show()

df_lite.ffill(inplace=True)
# print(df_lite[['Open','High','Low','Adj Close','Volume']][2:62].values.reshape(-1,5))

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

print(meanstd)