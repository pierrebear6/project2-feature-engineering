# FINAL ALGORITHM

from data import *
from svm_model import *
from rnn_model import *
import sys
import time
from rnn_evaluate import *
import tensorflow as tf

st = time.time()
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
    print('Using GPU')
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)
else:
    print('Using CPU')

# INITIALIZE PARAMETERS
ticker = 'amd'
window = 15
period = '6y'

# GATHER DATA
stock = GatherCandlestickData(ticker=ticker, period=period)
df1 = stock.import_data()
# df2 = stock.import_compare_data()

example_retest = False
if example_retest:
    df1 = pd.read_csv(rf'{filepath}/data/example_data/df1.csv')
    df2 = pd.read_csv(rf'{filepath}/data/example_data/df2.csv')

# PREPARE DATA
prep_data = PrepareData(ticker=ticker, df1=df1, period=period)
# compare = prep_data.compare_data(df2=df2)
df = prep_data.clean_data()

df = add_features(df=df)
df = add_svm_targets(features_df=df)
rnn_df = df.copy()

AnalyzeData(df=df, data_name=ticker).get_ydata_sumstats()
scaled_df = PrepareData(ticker=ticker, df1=df).minmaxscalar()
scaled_df['Target'] = scaled_df['Target'] * 2  # Convert to discrete label input
scaled_df = rfe_filter(df=scaled_df, label='Target', feedback=1)
X_train, y_train, X_test, y_test, X_val, y_val = PrepareData(ticker=ticker, df1=scaled_df).xy_split(target='Target',
                                                                                                    train_len=.7)

# TRAIN AND RUN SVM MODEL
svm_model = train_svm_model(X_train=X_train, y_train=y_train)
report = svm_model_predict(model=svm_model, X_test=X_test, y_test=y_test)
print(f'Number of test samples: {len(y_test)}')
print(report)

# TRAIN AND RUN RNN MODEL
window_size = 9
labels = ['relative_pct']

# RNN data prep
# df = df.drop(['Target'], axis=1)

# Simulate 'Target' variable data
# svm_accuracy = .5583
# data_to_sim = 1 - svm_accuracy
# num_rows = int(len(df) * data_to_sim)
# random_row_change = np.random.choice(df.index, num_rows, replace=False)
# sim_value = np.random.choice([0, 1, -1])
# df.loc[random_row_change, 'Target'] = sim_value

rnn_df_prep = PrepareData(ticker=ticker, df1=df)
unscaled_train, unscaled_val, unscaled_test = rnn_df_prep.split_data()

scaled_df = PrepareData(ticker=ticker, df1=df).minmaxscalar()
# scaled_df['Target'] = scaled_df['Target'] * 2  # Convert to discrete label input
scaled_df = rfe_filter(df=scaled_df, label='relative_pct', feedback=1)

train_df, val_df, test_df = PrepareData(ticker=ticker, df1=scaled_df).split_data()
X_train, y_train = create_model_inputs(train_df, window_size=window_size, labels=labels)
X_test, y_test = create_model_inputs(test_df, window_size=window_size, labels=labels)
X_val, y_val = create_model_inputs(val_df, window_size=window_size, labels=labels)

print(X_train.shape)

n_features = len(scaled_df.columns)
output_size = len(labels)

model = model(window_size=window_size, n_features=n_features, output_size=output_size)

# model = tuned_model(window_size=window_size, n_features=n_features, output_size=output_size, X_train=X_train,
#                     y_train=y_train, X_val=X_val, y_val=y_val)
model, mae = model_fit(model=model, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

predictions_df, forecast = rnn_model_predict(model=model, unscaled_df=df, labels=labels, X_test=X_test,
                                             window_size=window_size, unscaled_test=unscaled_test, n_future=1,
                                             original_df=df)

# PREDICTIONS
test_mae = performance(mae=mae, forecast=forecast, predictions_df=predictions_df, labels=labels,
                       original_df=df)
plot_results(labels=labels, predictions_df=predictions_df, model=' - v2')
plot_mae(mae=mae, model=' - v2')
store_results(predictions_df=predictions_df, ticker=ticker, labels=labels, test_mae=test_mae)

# SYSTEM OUTPUTS AND SAVES
et = time.time()
elapsed_time = et - st
print('\n')
print('Execution Time: ', elapsed_time, 'seconds', '\n', elapsed_time / 60, 'minutes')
