import datetime
from logging import getLogger, basicConfig, INFO
import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dropout, Dense
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from models import ModelMLP
from preprocessing import load_train_data, load_test_data
from submission import create_submission, create_nn_submission

from const import RESULT_DIR, PICKLE_DIR

logger = getLogger(__name__)


def create_model(input_shape, classes):
    cnn_model = Sequential()
    cnn_model.add(Conv1D(64, 3, activation='relu', input_shape=(input_shape, 1)))
    cnn_model.add(Conv1D(64, 3, activation='relu'))
    cnn_model.add(MaxPooling1D(3))
    cnn_model.add(Conv1D(128, 3, activation='relu'))
    cnn_model.add(Conv1D(128, 3, activation='relu'))
    cnn_model.add(GlobalAveragePooling1D())
    cnn_model.add(Dropout(0.5))
    cnn_model.add(Dense(classes, activation='softmax'))

    cnn_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['sparse_categorical_accuracy'])
    return cnn_model


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
        tr_x, va_x = x_train.iloc[train_idx], x_train.iloc[valid_idx]
        tr_y, va_y = y_train.iloc[train_idx], y_train.iloc[valid_idx]

        number_of_features = x_train.shape[1]
        classes = tr_y.nunique()

        model = create_model(number_of_features, classes)

        trn_x = np.array(tr_x).reshape((-1, number_of_features, 1))
        val_x = np.array(va_x).reshape((-1, number_of_features, 1))
        trn_y = np.array(tr_y).reshape((-1, 1))
        val_y = np.array(va_y).reshape((-1, 1))
        # trn_y = pd.get_dummies(tr_y).values.reshape(len(tr_y), classes)
        # val_y = pd.get_dummies(va_y).values.reshape(len(va_y), classes)
        # train_target = pd.get_dummies(df_train["open_channels"]).values.reshape(-1, 4000, 11)  # classification
        x_test = np.array(x_test).reshape((-1, number_of_features, 1))

        batch_size = 4096
        epochs = 5
        model.fit(trn_x, trn_y,
                  batch_size=batch_size, epochs=epochs,
                  verbose=1, validation_data=(val_x, val_y))
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
    pd.to_pickle(stacking_train, PICKLE_DIR + 'stacking_cnn_best_train.pkl')
    pd.to_pickle(stacking_test, PICKLE_DIR + 'stacking_cnn_best_test.pkl')

    return model


if __name__ == '__main__':
    log_format = '%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s '
    file_prefix = datetime.datetime.now().strftime('%m%d_%H%M')
    basicConfig(filename=RESULT_DIR + file_prefix + '_train_cnn.py.log', level=INFO, format=log_format)

    logger.info('##### Begin single model of cnn #####')

    trn_model = train_model()

    # 提出時のみコメントOFF
    create_nn_submission(trn_model)

    logger.info('##### single model of cnn END #####')



