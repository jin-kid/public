from logging import basicConfig, getLogger, INFO
import pandas as pd
from const import RESULT_DIR, PICKLE_DIR
from preprocessing import load_train_data, load_test_data
from submission import create_submission
import datetime
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from models import ModelLgbm

logger = getLogger(__name__)

# 2020/05/09
# CV max f1: 0.9360586378833148  / PL 0.880
# スタッキング用にベストパラメータをきつく調整した。過学習ぎみになると予想。
# 予想と違って過学習気味にならない。緩くした分類器と評価はあんまり変わらない。
# 個々の予測は変わってるが全体として変わってないのか、そもそも個々の予測が変わってないのか。
# 個々がかわってなければスタッキングしても意味ない。パラメータを8割で調整したの甘かった？


def train_model():
    x_train, y_train = load_train_data()
    x_test = load_test_data()

    sf = StratifiedKFold(n_splits=5,
                         shuffle=True,
                         random_state=0)

    params = {
        'objective': 'multiclass',
        'n_estimators': 10000,
        'silent': False,
        'metric': 'multi_logloss',
        'early_stopping_rounds': 100,
        'lambda_l1': 6.83751010327766,
        'lambda_l2': 7.991823346452801,
        'num_leaves': 164,
        'feature_fraction': 0.42,
        'bagging_fraction': 1.0,
        'bagging_freq': 0,
        'min_child_samples': 8,
        'seed': 0
    }
    pred_probas = []
    preds_test_probas = []
    va_idxes = []
    max_score = 0
    list_f1_score = []
    list_accuracy_score = []
    for train_idx, valid_idx in tqdm(list(sf.split(x_train, y_train))):
        trn_x = x_train.iloc[train_idx, :]
        val_x = x_train.iloc[valid_idx, :]

        trn_y = y_train[train_idx]
        val_y = y_train[valid_idx]

        model = ModelLgbm()
        model.fit(trn_x, trn_y, val_x, val_y, params)
        pred = model.predict(val_x)

        pred_proba = model.predict_proba(val_x)
        pred_probas.append(pred_proba)
        pred_test_proba = model.predict_proba(x_test)
        preds_test_probas.append(pred_test_proba)
        va_idxes.append(valid_idx)

        sc_f1 = f1_score(val_y, pred, average='macro')
        sc_accuracy = accuracy_score(val_y, pred)

        list_f1_score.append(sc_f1)
        list_accuracy_score.append(sc_accuracy)
        logger.info('     f1: {}, accuracy: {}'.format(sc_f1, sc_accuracy))

    f1_mean = np.mean(list_f1_score)
    accuracy_mean = np.mean(list_accuracy_score)
    logger.info('f1: {}, accuracy: {}'.format(f1_mean, accuracy_mean))

    if max_score < f1_mean:
        max_score = f1_mean

    logger.info('max f1: {}'.format(max_score))

    va_idxes = np.concatenate(va_idxes)
    pred_probas = np.concatenate(pred_probas, axis=0)
    order = np.argsort(va_idxes)
    stacking_train = pred_probas[order]
    stacking_test = np.mean(preds_test_probas, axis=0)
    pd.to_pickle(stacking_train, PICKLE_DIR + 'stacking_lightgbm_3_train.pkl')
    pd.to_pickle(stacking_test, PICKLE_DIR + 'stacking_lightgbm_3_test.pkl')
    return model


if __name__ == '__main__':
    log_format = '%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s '
    file_prefix = datetime.datetime.now().strftime('%m%d_%H%M')
    basicConfig(filename=RESULT_DIR + file_prefix + '_train_lightgbm_st3.py.log', level=INFO, format=log_format)

    logger.info('##### Begin single model of lightGBM #####')

    model = train_model()

    # 提出時のみコメントOFF
    create_submission(model)

    logger.info('##### single model of lightGBM END #####')

