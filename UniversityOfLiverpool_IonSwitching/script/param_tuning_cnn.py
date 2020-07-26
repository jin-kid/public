from logging import basicConfig, getLogger, INFO
import keras
import optuna
from keras import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Conv1D, MaxPooling1D, \
    MaxPool1D, GlobalAveragePooling1D
from optuna.integration import KerasPruningCallback
from preprocessing import load_train_data
import numpy as np
from sklearn.model_selection import train_test_split
import datetime
from const import RESULT_DIR

logger = getLogger(__name__)

# パラメータチューニングは実行時間がとてもかかる
# ある程度パラメータの候補を絞ってチューニングしないと、1日たっても終わらない
# 現実的に可能な範囲を検討してからチューニングプログラムを作ることにする

def create_model(n_layers, n_filter, kernel_size, activation, filter_step, mid_unit, dropout_rate, padding, classes, input_shape):
    model = Sequential()
    model.add(Conv1D(filters=n_filter, kernel_size=kernel_size, strides=filter_step,
                     activation=activation, padding=padding, input_shape=(input_shape, 1)))
    model.add(Conv1D(64, 3, activation=activation))
    model.add(MaxPooling1D(pool_size=kernel_size, strides=filter_step))
    model.add(Conv1D(128, 3, activation=activation))
    model.add(Conv1D(128, 3, activation=activation))
    model.add(GlobalAveragePooling1D())

    # for i in range(n_layers):
    #     model.add(Dense(mid_unit, activation=activation))
    #     model.add(Dropout(dropout_rate))

    model.add(Dropout(dropout_rate))
    model.add(Dense(classes, activation='softmax'))

    return model


def objective(trial):

    x_train, y_train = load_train_data()
    x_train_data, x_test, y_train_data, y_test, = train_test_split(x_train, y_train, test_size=0.2,
                                                                   random_state=0, shuffle=True, stratify=y_train)
    n_train = int(x_train_data.shape[0] * 0.75)
    x_trn, x_val = x_train_data[:n_train], x_train_data[n_train:]
    y_trn, y_val = y_train_data[:n_train], y_train_data[n_train:]

    # 前のセッションをクリアする
    keras.backend.clear_session()

    n_layers = trial.suggest_int('n_layer', 1, 3)
    n_filter = trial.suggest_int('n_filter', 5, 25)
    filter_step = trial.suggest_int('filter_step', 1, 10)
    kernel_size = trial.suggest_int('kernel_size', 1, 10)
    mid_units = int(trial.suggest_discrete_uniform('mid_units', 100, 500, 1))
    dropout_rate = trial.suggest_uniform('dropout_rate', 0, 1)
    activation = trial.suggest_categorical('activation', ['relu', 'sigmoid'])
    padding = trial.suggest_categorical('padding', ['valid', 'same', 'causal'])
    optimizer = trial.suggest_categorical('optimizer', ['sgd', 'Adam', 'RMSprop', 'Adadelta', 'Adagrad', 'Adamax'])
    batch_size = trial.suggest_categorical('batch_size', [1024, 2048, 4096])
    epochs = trial.suggest_int('epochs', 50, 200)
    # 学習モデルの構築と学習の開始
    model = create_model(n_layers, n_filter, kernel_size, activation, filter_step, mid_units, dropout_rate, padding, y_train_data.nunique(), x_train.shape[1])
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(x_trn, y_trn,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_val, y_val),
                        callbacks=[KerasPruningCallback(trial, 'val_acc')],
                        verbose=1)

    # 学習モデルの保存
    # model_json = model.to_json()
    # with open('keras_model.json', 'w') as f_model:
    #     f_model.write(model_json)
    # model.save_weights('keras_model.hdf5')

    # モデルの評価
    score = model.evaluate(x_test, y_test, verbose=0)
    logger.info('evaluate: {}'.format(score[1]))

    return -np.amax(history.history['val_accuracy'])


if __name__ == '__main__':
    log_format = '%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s '
    file_prefix = datetime.datetime.now().strftime('%m%d_%H%M')
    basicConfig(filename=RESULT_DIR + file_prefix + '_parameter_tuning_cnn.py.log', level=INFO, format=log_format)

    logger.info('##### begin parameter tuning #####')

    study = optuna.create_study(sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=100)
    logger.info('best_params : {}'.format(study.best_params))
    logger.info('best_value : {}'.format(-study.best_value))

    logger.info('\n --- sorted --- \n')
    sorted_best_params = sorted(study.best_params.items(), key=lambda x : x[0])
    for i, k in sorted_best_params:
        logger.info(i + ' : ' + str(k))

    pruned_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE]
    logger.info("Study statistics: ")
    logger.info("  Number of finished trials: ", len(study.trials))
    logger.info("  Number of pruned trials: ", len(pruned_trials))
    logger.info("  Number of complete trials: ", len(complete_trials))

    logger.info('##### parameter tuning end #####')
