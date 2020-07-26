import keras
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from logging import getLogger

logger = getLogger(__name__)


# lightGbmによるモデル
class ModelLgbm:

    def __init__(self):
        self.model = None
        logger.info('The model is LightGBM')

    def fit(self, tr_x, tr_y, va_x, va_y, params):
        logger.info('model parameter: {}'.format(params))
        dtrain = lgb.Dataset(tr_x, tr_y)
        dvalid = lgb.Dataset(va_x, va_y, reference=dtrain)
        logger.info('train x shape: {} / train y shape: {}'.format(tr_x.shape, tr_y.shape))
        logger.info('train x feature: {}'.format(tr_x.columns.values))
        params['num_class'] = tr_y.nunique()
        self.model = lgb.train(params, dtrain, valid_sets=[dtrain, dvalid])

    def predict(self, x):
        pred = self.model.predict(x)
        pred_max = np.argmax(pred, axis=1)
        fi = self.model.feature_importance()
        idx = np.argsort(fi)[::-1]
        logger.info('feature importance  :{}/{}'.format(x.columns.values[idx], fi[idx]))
        logger.info('num_feature         :{}'.format(self.model.num_feature()))
        return pred_max

    def predict_proba(self, x):
        pred = self.model.predict(x)
        fi = self.model.feature_importance()
        idx = np.argsort(fi)[::-1]
        logger.info('feature importance  :{}/{}'.format(x.columns.values[idx], fi[idx]))
        logger.info('num_feature         :{}'.format(self.model.num_feature()))
        return pred


# xgboostによるモデル
class ModelXgb:

    def __init__(self):
        self.model = None

    def fit(self, tr_x, tr_y, va_x, va_y, params):
        params = {'objective': 'binary:logistic', 'silent': 1, 'random_state': 71,
                  'eval_metric': 'logloss'}
        num_round = 10
        dtrain = xgb.DMatrix(tr_x, label=tr_y)
        dvalid = xgb.DMatrix(va_x, label=va_y)
        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        self.model = xgb.train(params, dtrain, num_round, evals=watchlist)

    def predict(self, x):
        data = xgb.DMatrix(x)
        pred = self.model.predict(data)
        return pred


# ニューラルネットによるモデル
class ModelMLP:

    def __init__(self):
        self.model = None
        self.scaler = None

    def fit(self, tr_x, tr_y, va_x, va_y, params):
        self.scaler = StandardScaler()
        self.scaler.fit(tr_x)

        batch_size = 128
        epochs = 10

        tr_x = self.scaler.transform(tr_x)
        va_x = self.scaler.transform(va_x)
        model = Sequential()
        model.add(Dense(256, activation='relu', input_shape=(tr_x.shape[1],)))
        model.add(Dropout(0.2))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(tr_y.nunique(), activation='softmax'))

        # opt = keras.optimizers.Adam(learning_rate=0.01)
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['sparse_categorical_accuracy'])

        print(np.array(tr_y).shape, np.array(va_y).shape)

        history = model.fit(tr_x, tr_y,
                            batch_size=batch_size, epochs=epochs,
                            verbose=1, validation_data=(va_x, va_y))
        self.model = model

    def predict(self, x):
        x = self.scaler.transform(x)
        pred = self.model.predict_classes(x).reshape(-1)
        return pred

    def predict_proba(self, x):
        x = self.scaler.transform(x)
        pred = self.model.predict_proba(x)
        return pred


# 線形モデル
class ModelLinear:

    def __init__(self):
        self.model = None
        self.scaler = None

    def fit(self, tr_x, tr_y, va_x, va_y, params):
        self.scaler = StandardScaler()
        self.scaler.fit(tr_x)
        tr_x = self.scaler.transform(tr_x)
        self.model = LogisticRegression(solver='lbfgs', C=1.0)
        self.model.fit(tr_x, tr_y)

    def predict(self, x):
        x = self.scaler.transform(x)
        pred = self.model.predict(x)
        return pred

    def predict_proba(self, x):
        x = self.scaler.transform(x)
        pred = self.model.predict_proba(x)[:, 1]
        return pred
