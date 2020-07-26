from logging import basicConfig, getLogger, INFO
from preprocessing import load_train_data
import optuna.integration.lightgbm as lgb
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import datetime
from const import RESULT_DIR

logger = getLogger(__name__)


def tuning_model(x, y):
    x_train_data, x_test, y_train_data, y_test, = train_test_split(x, y, test_size=0.2,
                                                                   random_state=0, shuffle=True, stratify=y)
    n_train = int(x_train_data.shape[0] * 0.75)
    x_trn, x_val = x_train_data[:n_train], x_train_data[n_train:]
    y_trn, y_val = y_train_data[:n_train], y_train_data[n_train:]

    trn = lgb.Dataset(x_trn, y_trn)
    val = lgb.Dataset(x_val, y_val)

    params = {'objective': 'multiclass',
              'num_class': y.nunique(),
              'metric': 'multi_logloss',
              'random_seed': 0}

    best_params = {}
    tuning_history = []

    booster = lgb.train(params,
                        trn,
                        valid_sets=val,
                        num_boost_round=10000,
                        early_stopping_rounds=100,
                        verbose_eval=200,
                        best_params=best_params,
                        tuning_history=tuning_history)

    # 調整後モデルで予測の実行
    train_pred = booster.predict(x_train_data)
    test_pred = booster.predict(x_test)

    fm_train = f1_score(y_train_data, np.argmax(train_pred, axis=1), average='macro')
    fm_test = f1_score(y_test, np.argmax(test_pred, axis=1), average='macro')
    ac_train = accuracy_score(y_train_data, np.argmax(train_pred, axis=1))
    ac_test = accuracy_score(y_test, np.argmax(test_pred, axis=1))

    # テストデータを用いた評価
    logger.info('best parameter: {}'.format(best_params))
    logger.info('tuning history: {}'.format(tuning_history))
    logger.info("train evaluation accuracy/f1 macro: {}/{}".format(ac_train, fm_train))
    logger.info("test  evaluation accuracy/f1 macro: {}/{}".format(ac_test, fm_test))


if __name__ == '__main__':
    log_format = '%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s '
    file_prefix = datetime.datetime.now().strftime('%m%d_%H%M')
    basicConfig(filename=RESULT_DIR + file_prefix + '_parameter_tuning_lightgbm.py.log', level=INFO, format=log_format)

    logger.info('parameter tuning start')

    x_train, y_train = load_train_data()
    tuning_model(x_train, y_train)

    logger.info('parameter tuning exit')
