import os
from logging import getLogger
import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from const import TRAIN_DATA, TEST_DATA, USE_PICKLE, PICKLE_DIR

logger = getLogger(__name__)


def read_csv(path):
    logger.debug('read csv entry')
    df = pd.read_csv(path)
    logger.debug('read csv exit')
    return df


def load_train_data():
    logger.debug('load train data entry')

    if USE_PICKLE:
        if os.path.exists(PICKLE_DIR + 'x_train.pkl'):
            logger.info('x_train read to pickle')
            return pd.read_pickle(PICKLE_DIR + 'x_train.pkl'), pd.read_pickle(PICKLE_DIR + 'y_train.pkl')

    logger.info('x_train created data')
    x_train, y_train = feature_exploration()
    logger.debug('number of shape after feature_exploration engineering: {}'.format(x_train.shape))
    # x_train, y_train = down_sampling(x_train, y_train)
    # logger.debug('number of shape after down sampling: {}'.format(x_train.shape))
    logger.debug('load train data exit')
    return x_train, y_train


def load_test_data():
    logger.debug('load test data entry')

    if USE_PICKLE:
        if os.path.exists(PICKLE_DIR + 'x_test.pkl'):
            logger.info('x_test read to pkl')
            return pd.read_pickle(PICKLE_DIR + 'x_test.pkl')

    logger.info('x_test created data')
    x_test = feature_exploration(is_train_data=False)
    logger.debug('number of shape after feature_exploration engineering: {}'.format(x_test.shape))
    logger.debug('load test data exit')
    return x_test


def load_submission_data():

    if USE_PICKLE:
        if os.path.exists(PICKLE_DIR + 'x_test.pkl'):
            logger.info('x_test read to pkl')
            return pd.read_pickle(PICKLE_DIR + 'x_test.pkl')

    return read_csv(TEST_DATA)


def feature_exploration(is_train_data=True, change_down_size=False, down_size=1000):

    df_train = pd.read_csv(TRAIN_DATA)
    df_test = pd.read_csv(TEST_DATA)

    if USE_PICKLE:
        if change_down_size:
            df_train = df_train.sample(n=down_size, random_state=0).sort_index()
            df_test = df_test.sample(n=down_size, random_state=0).sort_index()

    df = pd.concat([df_train, df_test])
    x_train = df[['time', 'signal']].copy()

    # scaler = StandardScaler()
    # scaler.fit(df)
    # df_std = pd.DataFrame(scaler.transform(df), columns=df.columns.values)
    # x_train['signal'] = df_std['signal']

    # yj = PowerTransformer(method='yeo-johnson')
    # yj.fit(x_train)
    # df_yj = pd.DataFrame(yj.transform(x_train), columns=x_train.columns.values)
    # x_train['signal'] = df_yj['signal']

    # p01 = x_train['signal'].quantile(0.01)
    # p99 = x_train['signal'].quantile(0.99)
    # x_train['signal'] = x_train['signal'].clip(p01, p99)

    x_train['time_bin'] = np.round(x_train['time'] / 50)
    x_train['time_bin2'] = np.round(x_train['time'] / 5)

    x_train['lag_t1'] = x_train.groupby('time_bin')['signal'].transform(lambda x: x.shift(1))
    x_train['lag_t2'] = x_train.groupby('time_bin')['signal'].transform(lambda x: x.shift(2))
    x_train['lag_t3'] = x_train.groupby('time_bin')['signal'].transform(lambda x: x.shift(3))
    x_train['lead_t1'] = x_train.groupby('time_bin')['signal'].transform(lambda x: x.shift(-1))
    x_train['lead_t2'] = x_train.groupby('time_bin')['signal'].transform(lambda x: x.shift(-2))
    x_train['lead_t3'] = x_train.groupby('time_bin')['signal'].transform(lambda x: x.shift(-3))

    for window in [1000, 5000, 10000, 20000, 40000, 80000]:
        # roll backwards
        x_train['signalmean_t' + str(window)] = x_train.groupby(['time_bin'])['signal'].transform(lambda x: x.shift(1).rolling(window).mean())
        x_train['signalstd_t' + str(window)] = x_train.groupby(['time_bin'])['signal'].transform(lambda x: x.shift(1).rolling(window).std())
        x_train['signalvar_t' + str(window)] = x_train.groupby(['time_bin'])['signal'].transform(lambda x: x.shift(1).rolling(window).var())
        x_train['signalmin_t' + str(window)] = x_train.groupby(['time_bin'])['signal'].transform(lambda x: x.shift(1).rolling(window).min())
        x_train['signalmax_t' + str(window)] = x_train.groupby(['time_bin'])['signal'].transform(lambda x: x.shift(1).rolling(window).max())
        min_max = (x_train['signal'] - x_train['signalmin_t' + str(window)]) / (x_train['signalmax_t' + str(window)] - x_train['signalmin_t' + str(window)])
        x_train['norm_t' + str(window)] = min_max * (np.floor(x_train['signalmax_t' + str(window)]) - np.ceil(x_train['signalmin_t' + str(window)]))

        # roll forward
        x_train['signalmean_t' + str(window) + '_lead'] = x_train.groupby(['time_bin'])['signal'].transform(lambda x: x.shift(- window - 1).rolling(window).mean())
        x_train['signalstd_t' + str(window) + '_lead'] = x_train.groupby(['time_bin'])['signal'].transform(lambda x: x.shift(- window - 1).rolling(window).std())
        x_train['signalvar_t' + str(window) + '_lead'] = x_train.groupby(['time_bin'])['signal'].transform(lambda x: x.shift(- window - 1).rolling(window).var())
        x_train['signalmin_t' + str(window) + '_lead'] = x_train.groupby(['time_bin'])['signal'].transform(lambda x: x.shift(- window - 1).rolling(window).min())
        x_train['signalmax_t' + str(window) + '_lead'] = x_train.groupby(['time_bin'])['signal'].transform(lambda x: x.shift(- window - 1).rolling(window).max())
        min_max = (x_train['signal'] - x_train['signalmin_t' + str(window) + '_lead']) / (x_train['signalmax_t' + str(window) + '_lead'] - x_train['signalmin_t' + str(window) + '_lead'])
        x_train['norm_t' + str(window) + '_lead'] = min_max * (np.floor(x_train['signalmax_t' + str(window) + '_lead']) - np.ceil(x_train['signalmin_t' + str(window) + '_lead']))

    for c in ['time_bin', 'time_bin2']:
        d = {}
        d['mean_' + c] = x_train.groupby([c])['signal'].mean()
        d['median_' + c] = x_train.groupby([c])['signal'].median()
        d['max_' + c] = x_train.groupby([c])['signal'].max()
        d['min_' + c] = x_train.groupby([c])['signal'].min()
        d['std_' + c] = x_train.groupby([c])['signal'].std()
        d['p10' + c] = x_train.groupby([c])['signal'].apply(lambda x: np.percentile(x, 10))
        d['p25' + c] = x_train.groupby([c])['signal'].apply(lambda x: np.percentile(x, 25))
        d['p75' + c] = x_train.groupby([c])['signal'].apply(lambda x: np.percentile(x, 75))
        d['p90' + c] = x_train.groupby([c])['signal'].apply(lambda x: np.percentile(x, 90))
        d['skew' + c] = x_train.groupby([c])['signal'].apply(lambda x: pd.Series(x).skew())
        d['kurtosis' + c] = x_train.groupby([c])['signal'].apply(lambda x: pd.Series(x).kurtosis())
        d['range_' + c] = d['max_' + c] - d['min_' + c]
        d['max_to_min_' + c] = d['max_' + c] / d['min_' + c]
        d['mean_abs_' + c] = x_train.groupby([c])['signal'].apply(lambda x: np.mean(np.abs(np.diff(x))))
        d['abs_max_' + c] = x_train.groupby([c])['signal'].apply(lambda x: np.max(np.abs(x)))
        d['abs_min_' + c] = x_train.groupby([c])['signal'].apply(lambda x: np.min(np.abs(x)))
        d['abs_avg_' + c] = (d['abs_max_' + c] + d['abs_min_' + c]) / 2
        for v in d:
            x_train[v] = x_train[c].map(d[v].to_dict())

    x_train = x_train.fillna(value=0)
    logger.debug('feature_exploration engineering train data shape :{}'.format(x_train.shape))
    logger.debug('feature_exploration engineering train data columns :{}'.format(x_train.columns.values))
    logger.debug('feature_exploration engineering train data last 5 :{}'.format(x_train.tail()))
    x_train = x_train.astype('float')

    if is_train_data:
        x_test = x_train.iloc[df_train.shape[0]:, :]
        x_test.reset_index(drop=True)
        x_train = x_train.iloc[:df_train.shape[0], :]
        x_train.reset_index(drop=True)
        logger.debug('feature_exploration engineering train x data last 5 :{}'.format(x_train.tail()))

        y_train = df_train['open_channels'].reset_index(drop=True)
        logger.debug('feature_exploration engineering train y data last 5 :{}'.format(y_train.tail()))
        y_train = y_train.astype('int')
        if USE_PICKLE:
            pd.to_pickle(x_train, PICKLE_DIR + 'x_train.pkl')
            pd.to_pickle(y_train, PICKLE_DIR + 'y_train.pkl')
            pd.to_pickle(x_test, PICKLE_DIR + 'x_test.pkl')

        return x_train, y_train

    x_test = x_train.iloc[df_train.shape[0]:, :].reset_index(drop=True)
    logger.debug('feature_exploration engineering test x data last 5 :{}'.format(x_test).tail())
    return x_test


def down_sampling(x_train, y_train):
    sampler = RandomUnderSampler(random_state=0)
    x_train, y_train = sampler.fit_resample(x_train, y_train)
    return x_train, y_train


if __name__ == '__main__':
    print(feature_exploration())
