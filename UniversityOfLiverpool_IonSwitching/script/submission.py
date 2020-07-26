from logging import getLogger

from const import SUBMIT_DIR
from preprocessing import load_test_data, load_submission_data
import numpy as np
import pandas as pd
import datetime

# SUBMIT_DIR= '../output/'

logger = getLogger(__name__)


def create_submission(model):

    x_test = load_test_data()
    test_feature = x_test.columns.values
    logger.info('test shape/feature_exploration: {}/{} '.format(test_feature.shape, test_feature))

    submission_test = x_test[test_feature]
    pred = model.predict(submission_test)

    sub = load_submission_data()

    main_key = np.array(sub["time"]).astype(float)
    file_prefix = datetime.datetime.now().strftime('%m%d_%H%M')
    submission = pd.DataFrame(pred, main_key, columns=['open_channels'])
    submission.to_csv(SUBMIT_DIR + file_prefix + '_submission.csv', index_label=['time'], float_format='%0.4f')


def create_mlp_submission(model):

    x_test = load_test_data()
    test_feature = x_test.columns.values
    logger.info('test shape/feature_exploration: {}/{} '.format(test_feature.shape, test_feature))

    submission_test = np.array(x_test[test_feature]).reshape(-1, x_test.shape[1])
    print(submission_test.shape)
    pred = np.argmax(model.predict(submission_test), axis=1)

    sub = load_submission_data()

    main_key = np.array(sub["time"]).astype(float)
    file_prefix = datetime.datetime.now().strftime('%m%d_%H%M')
    submission = pd.DataFrame(pred, main_key, columns=['open_channels'])
    submission.to_csv(SUBMIT_DIR + file_prefix + '_submission.csv', index_label=['time'], float_format='%0.4f')


def create_nn_submission(model):

    x_test = load_test_data()
    test_feature = x_test.columns.values
    logger.info('test shape/feature_exploration: {}/{} '.format(test_feature.shape, test_feature))

    submission_test = np.array(x_test[test_feature]).reshape(-1, x_test.shape[1], 1)
    print(submission_test.shape)
    pred = np.argmax(model.predict(submission_test), axis=1)

    sub = load_submission_data()

    main_key = np.array(sub["time"]).astype(float)
    file_prefix = datetime.datetime.now().strftime('%m%d_%H%M')
    submission = pd.DataFrame(pred, main_key, columns=['open_channels'])
    submission.to_csv(SUBMIT_DIR + file_prefix + '_submission.csv', index_label=['time'], float_format='%0.4f')