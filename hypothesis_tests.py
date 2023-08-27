# INFERENTIAL TEST OF DATA EXAMPLE

from data import *

time_range = 15
ticker = 'amd'
period = '5y'


def inference_features(df, window):
    """
    :param df: Df to be used to add features
    :param window: Used for rolling window calculations or time horizon calculations
    :return: Df with added features
    """
    df['Volatility'] = (df['Close'].pct_change().rolling(252).std() * (252 ** 0.5) * 100) / df['Close']
    df['Volatility_Pct'] = ((df['Volatility'] - df['Volatility'].rolling(window=window).min()) /
                            (df['Volatility'].rolling(window=window).max() - df['Volatility'].rolling(
                                window=window).min()))
    df['ATR'] = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=window).average_true_range()
    df['ATR_15'] = abs(df['Close'] - df['Close'].shift(-window))

    df['ATR_Diff'] = df['ATR_15'] / df['ATR']
    df['ATR_AvgDiff'] = df['ATR_Diff'].rolling(window=window).mean()

    df = df.dropna()
    df = df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1)
    return df


stock = GatherCandlestickData(ticker=ticker, period=period)
df1 = stock.import_data()
df2 = stock.import_compare_data()
prep_data = PrepareData(ticker=ticker, df1=df1, period=period)
prep_data.compare_data(df2=df2)
df = prep_data.clean_data()

df_features = inference_features(df=df, window=time_range)
print(df_features)

analyze_data = AnalyzeData(df=df_features, data_name='amd_inferential_test')
profile = analyze_data.get_ydata_sumstats()

# NO LINEAR CORRELATIONS - refer to ydata report
