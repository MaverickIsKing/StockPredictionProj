import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error
import features

df = pd.read_csv('NVDAGOOGTSLAMULITE2026-07-14-19-55-00.csv')
df_lite = df.loc[df['Ticker']=='LITE']
df_lite.reset_index(drop=True, inplace=True)

# df_log = features.log_change(df_lite,5)
# df_lite_ma5 = features.calculate_ma(df_lite,5)
# df_lite_pct_change60 = features.pct_change(df_lite,5)
# df_lite_ema30 = features.calculate_ema(df_lite, 30)
# pd.set_option('display.max_rows', None)
# print(df_lite_ema30)
df_lite.set_index('Date', inplace=True)
# print(df_lite)
# plt.plot(df_lite['Adj Close'])
# plt.show()

df_lite.ffill(inplace=True)
print(df_lite['Open','High','Low','Adj Close','Volumn'][2:62].values.reshape(-1,1))

def prepareDataForLSTM(df, label, memLength):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(df[label].values.reshape(-1, 1))
    data = []
    for i in range(len(df)-memLength):
        sequence = df[label][i:i+memLength].values.reshape(-1, 1)
        scaledSeq = scaler.transform(sequence)
        data.append(scaledSeq)
    return np.array(data)

# print(prepareDataForLSTM(df_lite, "Adj Close", 30))