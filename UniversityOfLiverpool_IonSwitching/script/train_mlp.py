import datetime
from logging import getLogger, basicConfig, INFO
import keras
import numpy as np
import pandas as pd
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, BatchNormalization
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from preprocessing import load_train_data, load_test_data
from submission import create_mlp_submission

from const import RESULT_DIR, PICKLE_DIR

logger = getLogger(__name__)

# 2020/05/09
# CV 0.929066482784491 / PL 0.028
# どのパラメータがダメ？オーバーフィッテイング？
# 課題：学習曲線を視覚化して確認する
# →本見てるとエポックもバッチサイズも隠れ層も全然足りない。もっと増やしてみる。
#   あとアーリーストッピングを追加
#
# 2020/05/13
# CV 0.9106014047680135 / PL 0.047
# もしかして実装がまずい？
# 本のサンプルみて確認した方がよさそう
#


def create_model(input_shape, classes):
    mlp_model = Sequential()
    mlp_model.add(Dense(96, activation='relu', input_shape=(input_shape,)))
    mlp_model.add(BatchNormalization())
    mlp_model.add(Dropout(0.2))
    mlp_model.add(Dense(96, activation='relu'))
    mlp_model.add(BatchNormalization())
    mlp_model.add(Dropout(0.2))
    mlp_model.add(Dense(96, activation='relu'))
    mlp_model.add(BatchNormalization())
    mlp_model.add(Dropout(0.2))
    mlp_model.add(Dense(96, activation='relu'))
    mlp_model.add(BatchNormalization())
    mlp_model.add(Dropout(0.2))
    mlp_model.add(Dense(classes, activation='softmax'))

    opt = keras.optimizers.Adam(learning_rate=0.0005)
    mlp_model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['sparse_categorical_accuracy'])
    return mlp_model


def train_model():
    x_train, y_train = load_train_data()
    x_test = load_test_data()

    sf = StratifiedKFold(n_splits=5,
                         shuffle=True,
                         random_state=0)

    params ={}
    pred_probas = []
    preds_test_probas = []
    va_idxes = []
    max_score = 0
    list_f1_score = list()
    list_accuracy_score = list()

    # 標準化はpreprocessingで実施する
    # 今回はココで
    df = pd.concat([x_train, x_test])
    scaler = StandardScaler()
    df_train = pd.DataFrame(scaler.fit_transform(df), columns=df.columns.values)
    x_train = df_train.iloc[:x_train.shape[0], :].reset_index(drop=True)
    x_test = df_train.iloc[x_train.shape[0]:, :].reset_index(drop=True)

    for train_idx, valid_idx in tqdm(list(sf.split(x_train, y_train))):
        trn_x, val_x = x_train.iloc[train_idx], x_train.iloc[valid_idx]
        trn_y, val_y = y_train.iloc[train_idx], y_train.iloc[valid_idx]

        number_of_features = x_train.shape[1]
        classes = trn_y.nunique()

        batch_size = 64
        epochs = 200
        patience = 20
        early_stopping = EarlyStopping(patience=patience, restore_best_weights=True)
        model = create_model(number_of_features, classes)
        history = model.fit(trn_x, trn_y,
                            batch_size=batch_size, epochs=epochs,
                            verbose=1, validation_data=(val_x, val_y),
                            callbacks=[early_stopping])
        logger.info('history : {}'.format(history))
        pred = np.argmax(model.predict(val_x), axis=1)

        pred_proba = model.predict_proba(val_x)
        pred_probas.append(pred_proba)
        pred_test_proba = model.predict_proba(x_test)
        preds_test_probas.append(pred_test_proba)
        va_idxes.append(valid_idx)

        sc_f1 = f1_score(val_y, pred, average='macro')
        sc_accuracy = accuracy_score(val_y, pred)

        list_f1_score.append(sc_f1)
        list_accuracy_score.append(sc_accuracy)
        logger.info('    f1: {}, accuracy: {}'.format(sc_f1, sc_accuracy))

    f1_mean = np.mean(list_f1_score)
    accuracy_mean = np.mean(list_accuracy_score)
    logger.info('mean f1: {}, accuracy: {}'.format(f1_mean, accuracy_mean))

    if max_score < f1_mean:
        max_score = f1_mean

    logger.info('max f1: {}'.format(max_score))

    va_idxes = np.concatenate(va_idxes)
    pred_probas = np.concatenate(pred_probas, axis=0)
    order = np.argsort(va_idxes)
    stacking_train = pred_probas[order]
    stacking_test = np.mean(preds_test_probas, axis=0)
    pd.to_pickle(stacking_train, PICKLE_DIR + 'stacking_mlp_best_train.pkl')
    pd.to_pickle(stacking_test, PICKLE_DIR + 'stacking_mlp_best_test.pkl')

    return model


if __name__ == '__main__':
    log_format = '%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s '
    file_prefix = datetime.datetime.now().strftime('%m%d_%H%M')
    basicConfig(filename=RESULT_DIR + file_prefix + '_train_mlp.py.log', level=INFO, format=log_format)

    logger.info('##### Begin single model of mlp #####')

    trn_model = train_model()

    # 提出時のみコメントOFF
    create_mlp_submission(trn_model)

    logger.info('##### single model of mlp END #####')



