import numpy as np
from numpy.linalg import norm

def gradient_decent(features, targets ,initial_state,step_size,precision,iterations = 1000):
    best_state = None
    min_val = float('inf')
    states_list = []
    n_samples = features.shape[0]

    min_val = float('inf')
    for s in step_size:
        for p in precision:
            cur_state = initial_state.copy()
            last_state = np.full_like(cur_state, float('inf'))
            iter = 0
            val = norm(cur_state - last_state)
            while val > p and iter < iterations :
                last_state = cur_state.copy()
                preds = predictions(features, cur_state)
                errs = errors(targets, preds)
                cur_state -= s * (1 / n_samples) * np.dot(features.T, errs)
                states_list.append(cur_state.copy())
                iter += 1

            cur_error = np.sum(np.abs(errors(targets, predictions(features, cur_state))))
            if cur_error < min_val:
                min_val = cur_error
                best_state = cur_state.copy()

    return best_state,states_list


def predictions (x, wights):
    return np.dot(x, wights)

def errors (y, y_pred):
    return y - y_pred