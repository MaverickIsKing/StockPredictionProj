import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
import matplotlib
matplotlib.use('Agg')

# For reading stock data from yahoo
import yfinance as yf

# For time stamps
from datetime import datetime



# Set up End and Start times for data grab
tech_list = ['NVDA', 'GOOG', 'TSLA', 'MU', 'LITE']

end = datetime.now()
start = datetime(end.year - 5, end.month, end.day)

def dataFetcher(list, start, end):
    # df_list = []
    # for ticker in list:
    #     data = yf.download(ticker, start, end, group_by="Ticker")
    #     data['ticker'] = ticker  # Add ticker column
    #     df_list.append(data)

    df = yf.download(list, start, end, group_by='Ticker', auto_adjust=False)
    return df.stack(level=0).rename_axis(['Date', 'Ticker']).reset_index(level=1)
    


data = dataFetcher(tech_list, start, end)

df = pd.DataFrame(data)
def fileGenerator(list, df):
    string = ''
    for item in list:
        string += item
    return df.to_csv(string + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') +'.csv')

fileGenerator(tech_list,df)

print(df)
# df_n = pd.DataFrame(NVDA)
# df_g = pd.DataFrame(GOOG)
# df_t = pd.DataFrame(TSLA)
# df_m = pd.DataFrame(MU)

# df_n.to_csv('NVDA.csv')
# df_g.to_csv('GOOG.csv')
# df_t.to_csv('TSLA.csv')
# df_m.to_csv('MU.csv')
# df = pd.concat([df_n, df_g, df_t, df_m])
# print(df_t.tail(10))

