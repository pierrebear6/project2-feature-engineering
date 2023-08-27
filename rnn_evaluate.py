import matplotlib
import matplotlib.pyplot as plt
import time
from datetime import datetime
import sys
import data

matplotlib.use('TkAgg')


def plot_results(labels, predictions_df, model):
    predictions_df = predictions_df[-60:]
    for n in labels:
        plt.plot(predictions_df['Date'], predictions_df[n], c='b', label=n)
        plt.plot(predictions_df['Date'], predictions_df['pred_' + n], c='r', label='pred_' + n)
        plt.legend(loc='upper right')
        plt.suptitle(datetime.now().strftime('%d-%m-%Y %H:%M:%S'))
        plt.title(n + model)
        plt.xticks(rotation=30)
        plt.show()


def plot_mae(mae, model):
    for n in mae.columns:
        plt.plot(mae.index, mae[n], label=n)
        plt.title(n + model)
        plt.xticks(rotation=30)
    plt.legend(loc='upper right')
    plt.ylabel('MAE')
    plt.suptitle(datetime.now().strftime('%d-%m-%Y %H:%M:%S'))
    plt.show()


def wall_time():
    """Showcases code to measure time to complete program"""
    st = time.time()
    # CODE GOES HERE
    et = time.time()
    elapsed_time = et - st
    print('\n')
    print('Execution Time: ', elapsed_time, 'seconds')


def performance(mae, forecast, predictions_df, labels, original_df):
    sum = 0
    test_mae = 0
    print('\n')
    original_df.set_index('Date', inplace=True)
    print(mae)
    print('\n')
    print(forecast)
    print('\n')
    for n in labels:
        predictions_df[n] = predictions_df[n] * original_df['relative_pct'][-len(predictions_df.index):]
        predictions_df['pred_' + n] = predictions_df['pred_' + n] * original_df['relative_pct'][
                                                                    -len(predictions_df.index):]
        forecast['pred_' + n] = forecast['pred_' + n] * original_df['relative_pct'].iloc[-1]
        for i in range(len(predictions_df.index)):
            sum += abs(predictions_df['pred_' + n][i] - predictions_df[n][i])
        test_mae = sum / len(predictions_df.index)
        print('MEAN ABSOLUTE ERROR(' + n + '): ', test_mae)
    print('\n')
    print(predictions_df)
    print('\n')
    print('forecast', forecast)

    return test_mae


def store_results(predictions_df, ticker, labels, test_mae):
    stdoutOrigin = sys.stdout
    dt = datetime.now().strftime('%m-%d-%Y')
    test_mae2 = round(test_mae, 3)
    sys.stdout = open(f"{filepath}{test_mae2}_{dt}_{ticker}.txt", "w")
    print('fart')
    sys.stdout.close()
    sys.stdout = stdoutOrigin
