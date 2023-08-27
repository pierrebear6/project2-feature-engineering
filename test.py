from data import *
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

# GATHER DATA
ticker = 'msft'
period = '6y'

g = GatherCandlestickData(ticker=ticker, period=period)
df1 = g.import_data()
prep = PrepareData(ticker='msft', df1=df1)
df1 = prep.clean_data()
df1['Date'] = pd.to_datetime(df1['Date'])

# FEATURES TO TEST
# Relative percent gain feature
df1['rolling_min'] = df1['Close'].rolling(window=30).min()
df1['rolling_max'] = df1['Close'].rolling(window=30).max()
df1['relative_pct'] = (df1['rolling_max'] - df1['Close']) / (
        df1['rolling_max'] - df1['rolling_min'])

# Log returns feature
df1['log_return'] = np.log10(df1['Close'] / df1['Close'].shift())
df1 = df1.dropna()

# ADF test for stationarity
result = adfuller(df1['relative_pct'])
print('p-value for relative_pct:', result[1])
result = adfuller(df1['log_return'])
print('p-value for log_return:', result[1])

df1 = df1[-250:]
plt.figure(figsize=(10, 6))
plt.acorr(df1['relative_pct'], maxlags=25)
plt.xlim(0, 25)
plt.title('relative_pct autocorrelation - maxlags=25')

plt.figure(figsize=(10, 6))
plt.acorr(df1['log_return'], maxlags=25)
plt.xlim(0, 25)
plt.title('log_return autocorrelation - maxlags=25')

plt.figure(figsize=(10, 6))
plt.plot(df1['Date'], df1['relative_pct'])
plt.title('relative_pct over time')

plt.figure(figsize=(10, 6))
plt.plot(df1['Date'], df1['log_return'])
plt.title('log_return over time')
plt.show()