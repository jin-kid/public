import datetime
from logging import getLogger, basicConfig, INFO
import pandas as pd
import lightgbm as lgb
from lightgbm import Dataset
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, ParameterGrid, train_test_split
from tqdm import tqdm

from const import PICKLE_DIR, RESULT_DIR, SUBMIT_DIR
from evaluation import evaluate_macroF1_lgb, macro_f1_score
from models import ModelLgbm, ModelMLP, ModelLinear
from preprocessing import load_train_data, load_test_data, load_submission_data
from submission import create_submission

logger = getLogger(__name__)


def predict_cv(model, train_x, train_y, test_x, params):
    preds = []
    preds_test = []
    va_idxes = []

    sf = StratifiedKFold(n_splits=5,
                         shuffle=True,
                         random_state=0)
    list_evaluation = list()
    # list_f1_score = list()
    # list_accuracy_score = list()

    for tr_idx, va_idx in tqdm(list(sf.split(train_x, train_y))):
        tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
        tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

        model.fit(tr_x, tr_y, va_x, va_y, params)
        # pred = model.predict(va_x)
        pred = model.predict_proba(va_x)
        preds.append(pred)
        pred_test = model.predict_proba(test_x)
        preds_test.append(pred_test)
        va_idxes.append(va_idx)

        sc_evaluation = log_loss(va_y, pred, eps=1e-7)
        # sc_f1 = f1_score(va_y, pred, average='macro', labels=tr_y.unique())
        # sc_accuracy = accuracy_score(va_y, pred)

        list_evaluation.append(sc_evaluation)
        # list_f1_score.append(sc_f1)
        # list_accuracy_score.append(sc_accuracy)
        # logger.info('    f1: {}, accuracy: {}'.format(sc_f1, sc_accuracy))
        logger.info('    log loss: {}'.format(sc_evaluation))

    # f1_mean = np.mean(list_f1_score)
    # accuracy_mean = np.mean(list_accuracy_score)
    # logger.info('mean f1: {}, accuracy: {}'.format(f1_mean, accuracy_mean))
    evaluation_mean = np.mean(list_evaluation)
    logger.info('log loss: {}'.format(evaluation_mean))

    va_idxes = np.concatenate(va_idxes)
    preds = np.concatenate(preds, axis=0)
    order = np.argsort(va_idxes)
    pred_train = preds[order]

    preds_test = np.mean(preds_test, axis=0)
    return pred_train, preds_test


def stacking_process():
    train_x, train_y = load_train_data()
    test_x = load_test_data()
    logger.info('[stacking_process 1 layer] lightGBM start')
    model1 = ModelLgbm()
    model1_params = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'n_estimators': 10000,
        'early_stopping_rounds': 100,
        'learning_rate': 0.1,
        'max_depth': 3,
        'min_child_weight': 0.001,
        'min_data_in_leaf': 3,
        'num_leaves': 10,
        'colsample_bytree': 0.9,
        'reg_alpha': 1,
        'reg_lambda': 0.1,
        'seed': 0
    }
    pred_train_1, pred_test_1 = predict_cv(model1, train_x, train_y, test_x, model1_params)
    logger.info('[stacking_process 1 layer] lightGBM end')

    logger.info('[stacking_process 2 layer] lightGBM start')
    model2 = ModelLgbm()
    model2_params = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'n_estimators': 10000,
        'early_stopping_rounds': 100,
        'learning_rate': 0.1,
        'max_depth': 5,
        'min_child_weight': 0.001,
        'min_data_in_leaf': 3,
        'num_leaves': 10,
        'colsample_bytree': 0.9,
        'reg_alpha': 1,
        'reg_lambda': 0.1,
        'seed': 0
    }
    pred_train_2, pred_test_2 = predict_cv(model2, train_x, train_y, test_x, model2_params)
    logger.info('[stacking_process 2 layer] lightGBM end')

    logger.info('[stacking_process 3 layer] lightGBM start')
    model3 = ModelLgbm()
    model3_params = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'n_estimators': 10000,
        'early_stopping_rounds': 100,
        'learning_rate': 0.1,
        'max_depth': 10,
        'min_child_weight': 0.001,
        'min_data_in_leaf': 3,
        'num_leaves': 10,
        'colsample_bytree': 0.9,
        'reg_alpha': 1,
        'reg_lambda': 0.1,
        'seed': 0
    }
    pred_train_3, pred_test_3 = predict_cv(model3, train_x, train_y, test_x, model3_params)
    logger.info('[stacking_process 3 layer] lightGBM end')

    logger.info('[stacking_process 4 layer] MLP start')
    model4 = ModelMLP()
    model4_params = {}
    pred_train_4, pred_test_4 = predict_cv(model4, train_x, train_y, test_x, model4_params)
    logger.info('[stacking_process 4 layer] MLP end')

    # 1層目モデル評価
    logger.info(f'logloss: {log_loss(train_y, pred_train_1, eps=1e-7):.4f}')
    logger.info(f'logloss: {log_loss(train_y, pred_train_2, eps=1e-7):.4f}')
    logger.info(f'logloss: {log_loss(train_y, pred_train_3, eps=1e-7):.4f}')
    logger.info(f'logloss: {log_loss(train_y, pred_train_4, eps=1e-7):.4f}')
    # logger.info(f'f1 macro: {f1_score(train_y, pred_train_1, average="macro", labels=train_y.unique()):.4f}')
    # logger.info(f'f1 macro: {f1_score(train_y, pred_train_2, average="macro", labels=train_y.unique()):.4f}')
    # logger.info(f'f1 macro: {f1_score(train_y, pred_train_3, average="macro", labels=train_y.unique()):.4f}')
    # logger.info(f'f1 macro: {f1_score(train_y, pred_train_4, average="macro", labels=train_y.unique()):.4f}')

    # 予測値を特徴量としてデータフレームを作成
    train_x_2 = pd.DataFrame(np.hstack([pred_train_1, pred_train_2, pred_train_3, pred_train_4]))
    test_x_2 = pd.DataFrame(np.hstack([pred_test_1, pred_test_2, pred_test_3, pred_test_4]))
    # train_x_2 = pd.DataFrame({'pred_1': pred_train_1, 'pred_2': pred_train_2,
    #                           'pred_3': pred_train_3, 'pred_4': pred_train_4})
    # test_x_2 = pd.DataFrame({'pred_1': pred_test_1, 'pred_2': pred_test_2,
    #                          'pred_3': pred_test_3, 'pred_4': pred_test_4})

    # 2層目のモデル
    # pred_train_2は、2層目のモデルの学習データのクロスバリデーションでの予測値
    # pred_test_2は、2層目のモデルのテストデータの予測値
    model_l2 = ModelLinear()
    model_l2_params = {}
    pred_train_l2, pred_test_l2 = predict_meta_cv(model_l2, train_x_2, train_y, test_x_2, model_l2_params)
    logger.info(f'f1 macro: {f1_score(train_y, pred_train_l2, average="macro", labels=train_y.unique()):.4f}')

    return model_l2


def predict_meta_cv(mt_model, trn_x, test_x, param):
    tmp_x, trn_y = load_train_data()
    sf = StratifiedKFold(n_splits=5,
                         shuffle=True,
                         random_state=0)
    list_f1_score = list()
    list_accuracy_score = list()

    for tr_idx, va_idx in tqdm(list(sf.split(trn_x, trn_y))):
        tr_x, va_x = trn_x.iloc[tr_idx], trn_x.iloc[va_idx]
        tr_y, va_y = trn_y.iloc[tr_idx], trn_y.iloc[va_idx]

        mt_model.fit(tr_x, tr_y)
        pred = mt_model.predict(va_x)

        sc_f1 = f1_score(va_y, pred, average='macro', labels=tr_y.unique())
        sc_accuracy = accuracy_score(va_y, pred)

        list_f1_score.append(sc_f1)
        list_accuracy_score.append(sc_accuracy)
        logger.info('    f1: {}, accuracy: {}'.format(sc_f1, sc_accuracy))

    f1_mean = np.mean(list_f1_score)
    accuracy_mean = np.mean(list_accuracy_score)
    logger.info('mean f1: {}, accuracy: {}'.format(f1_mean, accuracy_mean))

    preds_test = mt_model.predict(test_x)
    return preds_test


def meta_model_linear():
    linear_model = LogisticRegression(solver='lbfgs', C=1.0)
    return linear_model


def stacking_with_pickle():
    pred_train_1 = pd.read_pickle(PICKLE_DIR + 'stacking_lightgbm_best_train.pkl')
    pred_test_1 = pd.read_pickle(PICKLE_DIR + 'stacking_lightgbm_best_test.pkl')
    pred_train_2 = pd.read_pickle(PICKLE_DIR + 'stacking_lightgbm_2_train.pkl')
    pred_test_2 = pd.read_pickle(PICKLE_DIR + 'stacking_lightgbm_2_test.pkl')
    pred_train_3 = pd.read_pickle(PICKLE_DIR + 'stacking_lightgbm_3_train.pkl')
    pred_test_3 = pd.read_pickle(PICKLE_DIR + 'stacking_lightgbm_3_test.pkl')
    pred_train_4 = pd.read_pickle(PICKLE_DIR + 'stacking_mlp_best_train.pkl')
    pred_test_4 = pd.read_pickle(PICKLE_DIR + 'stacking_mlp_best_test.pkl')
    pred_train_5 = pd.read_pickle(PICKLE_DIR + 'stacking_cnn_best_train.pkl')
    pred_test_5 = pd.read_pickle(PICKLE_DIR + 'stacking_cnn_best_test.pkl')

    x_trn = pd.DataFrame(np.concatenate((pred_train_1, pred_train_2, pred_train_3, pred_train_4, pred_train_5), axis=1))
    x_test = pd.DataFrame(np.concatenate((pred_test_1, pred_test_2, pred_test_3, pred_test_4, pred_test_5), axis=1))

    return x_trn, x_test


def create_stacking_submission(t_pred):

    sub = load_submission_data()
    main_key = np.array(sub["time"]).astype(float)
    submit_file_prefix = datetime.datetime.now().strftime('%m%d_%H%M')
    submission = pd.DataFrame(t_pred, main_key, columns=['open_channels'])
    submission.to_csv(SUBMIT_DIR + submit_file_prefix + '_submission.csv', index_label=['time'], float_format='%0.4f')


if __name__ == '__main__':
    log_format = '%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s '
    file_prefix = datetime.datetime.now().strftime('%m%d_%H%M')
    basicConfig(filename=RESULT_DIR + file_prefix + '_train_lightgbm.py.log', level=INFO, format=log_format)

    logger.info('##### Begin train stacking #####')

    train_x, test_x = stacking_with_pickle()
    meta_model = meta_model_linear()
    params = {}

    pred_test = predict_meta_cv(meta_model, train_x, test_x, params)

    # 提出時のみコメントOFF
    create_stacking_submission(pred_test)

    logger.info('##### train stacking END #####')
