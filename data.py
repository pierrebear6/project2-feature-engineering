# IMPORT, CLEAN AND PREP DATA

import yfinance as yf
import requests
from sklearn.preprocessing import MinMaxScaler
import math
from ydata_profiling import ProfileReport
from ta.volatility import AverageTrueRange
from functions.feature_stationarity import *
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
import statsmodels.api as sm

filepath = '#######'


class GatherCandlestickData():
    def __init__(self, ticker=None, interval='1d', period='6y', output='full'):
        """
        :param ticker: stock ticker
        :param interval: time interval
        :param period: time period
        :param output: conditional - to return df output (print)
        """
        self.ticker = ticker
        self.interval = interval
        self.period = period
        self.output = output

    def import_data(self):
        """
        :return: yfin_data: stock data from yfinance
        """
        yr = int(self.period[0]) + 1
        period = str(yr) + self.period[1]
        yfin_data = yf.Ticker(self.ticker).history(period=period, interval=self.interval)
        date_trim = int(self.period[0]) * 252
        yfin_data = yfin_data[-date_trim:]
        yfin_data.reset_index(inplace=True)
        yfin_data['Date'] = pd.to_datetime(yfin_data['Date']).dt.strftime('%Y-%m-%d')
        yfin_data = yfin_data.drop(['Stock Splits', 'Dividends'], axis=1)

        def roundup(number):
            return math.ceil(number * 100) / 100

        yfin_data[['Open', 'High', 'Low', 'Close']] = yfin_data[['Open', 'High', 'Low', 'Close']].apply(
            lambda x: x.apply(roundup))
        yfin_data.reset_index(drop=True)

        return yfin_data

    def import_compare_data(self):
        """
        Note: Results in compilation of stock data to check with another data source. Separate function due to API
        request limits
        :return: alpha_data: stock data from alpha vantage
        """
        # output size can also be set to 'full'
        url = "https://alpha-vantage.p.rapidapi.com/query"
        querystring = {"function": "TIME_SERIES_DAILY", "symbol": self.ticker, "outputsize": 'full', "datatype": "json"}
        headers = {
            "X-RapidAPI-Key": "ad9bd43ad3msh85edfc8412a4837p1eb8acjsn625bfe4be209",
            "X-RapidAPI-Host": "alpha-vantage.p.rapidapi.com"
        }
        response = requests.request("GET", url, headers=headers, params=querystring)
        json = response.json()
        alpha_data = pd.DataFrame(json['Time Series (Daily)']).transpose()
        alpha_data = alpha_data.iloc[::-1]
        date_trim = int(self.period[0]) * 252
        alpha_data = alpha_data[-date_trim:]
        alpha_data.reset_index(inplace=True)
        alpha_data = alpha_data.rename(columns={'index': 'Date'})
        alpha_data.rename(
            columns={'1. open': 'Open', '2. high': 'High', '3. low': 'Low', '4. close': 'Close', '5. volume': 'Volume'},
            inplace=True)
        alpha_data[['Open', 'High', 'Low', 'Volume', 'Close']] = alpha_data[
            ['Open', 'High', 'Low', 'Volume', 'Close']].apply(
            pd.to_numeric)

        def roundup(number):
            return math.ceil(number * 100) / 100

        alpha_data[['Open', 'High', 'Low', 'Close']] = alpha_data[['Open', 'High', 'Low', 'Close']].apply(
            lambda x: x.apply(roundup))

        alpha_data.to_csv(f'{filepath}/data/imports/{self.ticker}_alpha.csv', index=False)
        return alpha_data


class PrepareData():
    def __init__(self, ticker, df1, tolerance=.05, period='6y', replace_zero=False):
        """
        :param ticker: stock symbol
        :param df1: primary df
        :param tolerance: tolerance for comparison error between datasets
        :param period: length of historical data
        :param replace_zero: bool variable to determine if zero values should be imputed
        """
        self.df1 = df1
        self.tolerance = tolerance
        self.period = period
        self.ticker = ticker
        self.replace_zero = replace_zero

    def compare_data(self, df2):
        """
        :param df2: secondary df used to compare primary df with
        :return: Feedback on if the data sources are consistent
        """
        df1 = self.df1[['Open', 'High', 'Low', 'Close', 'Volume']]
        df2 = df2[['Open', 'High', 'Low', 'Close', 'Volume']]
        tolerance = self.tolerance
        # Check if the DataFrames have the same shape
        if df1.shape != df2.shape:
            print('Dataframe shapes do not match!')
            return False

        diff = np.abs(df1 - df2)
        max_tolerance = np.max(np.abs(df1) * tolerance)

        # Check if all values are similar within the tolerance
        are_similar = np.all(diff <= max_tolerance)

        # Return the rows where DataFrames are not similar
        if not are_similar:
            percent_diff = (diff / df1) * 100
            rows_diff = percent_diff[~np.all(diff <= max_tolerance, axis=1)]
            # print(rows_diff)
            print('Data source not consistent!')

            return False
        else:
            print('Data source consistent!')
            return True

    def clean_data(self):
        """
        :return: Either a clean df or feedback on why data is faulty
        """
        df = self.df1
        period = int(self.period[0])
        ticker = self.ticker
        replace_zero = self.replace_zero

        try:
            fill = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in fill:
                if replace_zero:
                    mask = df[col] == 0
                    df[col] = df[col].where(~mask, (df[col].shift(-1) + df[col].shift(1)) / 2)
            data_small = df.shape[0] < period * 251
            nan_present = df.isna().values.any()
            zero_present = (df == 0).any().any()
            empty_df = df.empty

            text_map = {data_small: 'The dataset is too small.', nan_present: 'NaN/Null values are present',
                        zero_present: 'Zeroes are present', empty_df: 'The dataset is empty'}
            dirty = False
            for var, text in text_map.items():
                if var:
                    print(text)
                    dirty = True
            if not dirty:
                df.to_csv(rf'{filepath}\data\clean_data\{ticker}_clean.csv', index=False)
                print('Clean data stored!')
                return df
        except KeyError as e:
            error_message = str(e)
            if "1d data not available" or "not found in axis" in error_message:
                print("Data not available for the specified time range. Skipping operation.")
            else:
                print("An error occurred:", error_message)

    def split_data(self, train_len=.7, test_len=.2):
        """
        :param test_len:
        :param train_len: Proportion of data that is assigned to the training dataset
        :return: train and test dfs
        """
        df = self.df1
        n = len(self.df1)
        train_df = df[0:int(n * train_len)]
        test_df = df[int(n * (train_len)):int(n * (train_len + test_len))]
        val_df = df[int(n * (train_len + test_len)):]
        return train_df, test_df, val_df

    def minmaxscalar(self):
        """
        :return: Minmax scaled df
        """
        data = self.df1
        data = data.drop(['Date'], axis=1)
        col_names = data.columns
        scaler = MinMaxScaler().fit(data.values)
        scaled_data_np = scaler.transform(data.values)
        scaled_data_df = pd.DataFrame(scaled_data_np, columns=col_names)
        return scaled_data_df

    def inverse_scale_data(self, scaled_data):
        """
        :param scaled_data: Minmax scaled data
        :return: inverse scaled data
        """
        data = self.df1
        col_names = scaled_data.columns
        scaler = MinMaxScaler().fit(data.values)
        inverse_scaled_np = scaler.inverse_transform(scaled_data)
        inverse_scaled_df = pd.DataFrame(inverse_scaled_np, columns=col_names)
        return inverse_scaled_df

    def xy_split(self, target, train_len=.7, test_len=.2):
        """
        :param test_len:
        :param target: Target column to be used as label
        :param train_len: Proportion of data that is assigned to the training dataset
        :return: Train and test splits
        """
        df = self.df1
        n = len(self.df1)
        train_df = df[0:int(n * train_len)]
        test_df = df[int(n * (train_len)):int(n * (train_len + test_len))]
        val_df = df[int(n * (train_len + test_len)):]
        X_train = train_df.drop(target, axis=1).values
        y_train = train_df[target].values
        X_test = test_df.drop(target, axis=1).values
        y_test = test_df[target].values
        X_val = val_df.drop(target, axis=1).values
        y_val = val_df[target].values
        return X_train, y_train, X_test, y_test, X_val, y_val


class AnalyzeData():
    def __init__(self, df, data_name):
        self.df = df
        self.data_name = data_name

    def get_pd_sumstats(self):
        """
        :return: None. Prints basic pandas description statistics
        """
        df = self.df
        print(df.describe())

    def get_ydata_sumstats(self):
        """
        :return: Statistical report on df
        """
        df = self.df
        data_name = self.data_name
        profile = ProfileReport(df, title=data_name)
        profile.to_file(rf'{filepath}\data\summary_stats\{data_name}_stats_report.html')
        return df


def rfe_filter(df, label, feedback):
    """ Returns filtered features using RFE filter method """
    # Data
    data = df
    df = data

    # Filter
    lr = LinearRegression()
    rfe = RFE(estimator=lr, n_features_to_select=15)
    rfe.fit(df, df[label])
    rfe.support_
    rfe.ranking_
    Columns = df.columns
    RFE_support = rfe.support_
    RFE_ranking = rfe.ranking_
    dataset = pd.DataFrame({'Columns': Columns, 'RFE_support': RFE_support, 'RFE_ranking': RFE_ranking},
                           columns=['Columns', 'RFE_support', 'RFE_ranking'])
    selected = dataset[(dataset["RFE_support"] == True) & (dataset["RFE_ranking"] == 1)]
    filtered_features = selected['Columns'].values
    remove = []
    cols = list(df)[1:]
    n = 0
    for n in cols:
        if n not in filtered_features:
            remove.append(n)

    if feedback == 1:
        print('\nORIGINAL FEATURES:')
        print(cols)
        print('\nFILTERED FEATURES:')
        print(filtered_features)
        print('\nREMOVED COLUMNS:')
        print(remove)
        df = data.drop(remove, axis=1)
        cols = df.columns
        print('\nHEAD:')
        print(df.head())
    else:
        pass
    return df


def add_features(df):
    features_df = df.copy()

    # BASE STOCK INFO
    features_df['overnight_pctgain'] = ((features_df['Open'] - features_df['Close'].shift()) / features_df[
        'Close'].shift()) * 100
    features_df['intraday_pctgain'] = ((features_df['Close'] - features_df['Open']) / features_df['Open']) * 100
    features_df['realized_volatility'] = (features_df['Close'].pct_change().rolling(252).std() * (252 ** 0.5) * 100) / \
                                         features_df[
                                             'Close']
    features_df['log_return'] = np.log10(features_df['Close'] / features_df['Close'].shift())
    features_df['pct_return'] = features_df['Close'].pct_change()
    rfr = 0  # risk-free rate
    features_df['sharpe_ratio'] = (features_df['pct_return'].mean() - rfr) / features_df['pct_return'].std()
    # FLUCTUATIONS
    dec_ts_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for dec_ts_col in dec_ts_columns:
        features_df[f'residual_{dec_ts_col}'] = decompose_time_series(df=features_df, column_name=dec_ts_col)

    # BASE STATS
    features_df['rolling_min'] = features_df['Close'].rolling(window=30).min()
    features_df['rolling_max'] = features_df['Close'].rolling(window=30).max()
    features_df['relative_pct'] = (features_df['rolling_max'] - features_df['Close']) / (
            features_df['rolling_max'] - features_df['rolling_min'])
    features_df = features_df.drop(['rolling_max', 'rolling_min'], axis=1)

    # Volatility gauges
    features_df['rolling_min'] = features_df['realized_volatility'].rolling(window=30).min()
    features_df['rolling_max'] = features_df['realized_volatility'].rolling(window=30).max()
    features_df['relative_pct_volatility'] = (features_df['rolling_max'] - features_df['realized_volatility']) / (
            features_df['rolling_max'] - features_df['rolling_min'])
    features_df = features_df.drop(['rolling_max', 'rolling_min'], axis=1)

    features_df['ATR'] = AverageTrueRange(high=features_df['High'], low=features_df['Low'], close=features_df['Close'],
                                          window=9).average_true_range() / \
                         features_df['Close']  # Non-stationary? Maybe remove
    features_df['rolling_min'] = features_df['ATR'].rolling(window=30).min()
    features_df['rolling_max'] = features_df['ATR'].rolling(window=30).max()
    features_df['relative_pct_atr'] = (features_df['rolling_max'] - features_df['ATR']) / (
            features_df['rolling_max'] - features_df['rolling_min'])
    features_df = features_df.drop(['rolling_max', 'rolling_min'], axis=1)
    # OTHER TIME SERIES COMPONENTS
    # Elementary trend change point feature
    features_df['ema3'] = features_df['Close'].ewm(span=3, adjust=False).mean()
    features_df['ema9'] = features_df['Close'].ewm(span=9, adjust=False).mean()
    features_df['ema_changepoint'] = (features_df['ema3'] > features_df['ema9']).astype(int)

    # Elementary volatility change point feature
    features_df['atr_ema9'] = features_df['ATR'].ewm(span=9, adjust=False).mean()
    features_df['atr_changepoint'] = (features_df['ATR'] > features_df['atr_ema9']).astype(int)

    return features_df


def add_svm_targets(features_df):
    svm_df = features_df.copy()
    svm_df['ema3_9'] = svm_df['ema3'].shift(-9)
    svm_df['ema9_9'] = svm_df['ema9'].shift(-9)
    svm_df['atr_ema9_9'] = svm_df['atr_ema9'].shift(-9)
    svm_df['ATR_9'] = svm_df['ATR'].shift(-9)

    svm_df['Target'] = 0
    svm_df.loc[(svm_df['ema9_9'] > svm_df['ema3_9']) & (svm_df['atr_ema9_9'] < svm_df['ATR_9']), 'Target'] = -1
    svm_df.loc[(svm_df['ema9_9'] < svm_df['ema3_9']) & (svm_df['atr_ema9_9'] < svm_df['ATR_9']), 'Target'] = 1

    svm_df = svm_df.drop(['ema3_9', 'ema9_9', 'atr_ema9_9', 'ATR_9'], axis=1)
    svm_df = svm_df.dropna()
    return svm_df


def create_model_inputs(df, window_size, labels):
    X = []
    y = []
    X_as_np = df.to_numpy()
    y_as_np = df[labels].to_numpy()
    for i in range(len(X_as_np) - window_size):
        row = [a for a in X_as_np[i:i + window_size]]
        X.append(row)
        label = y_as_np[i + window_size]
        y.append(label)
    return np.array(X), np.array(y)
