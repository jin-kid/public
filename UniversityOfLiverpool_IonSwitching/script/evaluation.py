from sklearn.metrics import f1_score
import numpy as np
import numba as nb


def evaluate_macroF1_lgb(truth, predictions):
    # this follows the discussion in https://github.com/Microsoft/LightGBM/issues/1483
    pred_labels = predictions.reshape(len(np.unique(truth)), -1).argmax(axis=0)
    f1 = f1_score(truth, pred_labels, average='macro')
    return 'macroF1', f1, True


@nb.jit(nb.float64(nb.int32[:], nb.int32[:], nb.int64), nopython=True, nogil=True)
def macro_f1_score(y_true, y_pred, n_labels):
    total_f1 = 0.
    for i in range(n_labels):
        yt = y_true == i
        yp = y_pred == i

        tp = np.sum(yt & yp)

        tpfp = np.sum(yp)
        tpfn = np.sum(yt)
        if tpfp == 0:
            print('[WARNING] F-score is ill-defined and being set to 0.0 in labels with no predicted samples.')
            precision = 0.
        else:
            precision = tp / tpfp
        if tpfn == 0:
            print(f'[ERROR] label not found in y_true...')
            recall = 0.
        else:
            recall = tp / tpfn

        if precision == 0. or recall == 0.:
            f1 = 0.
        else:
            f1 = 2 * precision * recall / (precision + recall)
        total_f1 += f1
    return total_f1 / n_labels
