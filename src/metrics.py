
import numpy as np
import numba


@numba.jit(nopython=True)
def _is_comparable(t_i, t_j, d_i, d_j):
    return ((t_i < t_j) & d_i) | ((t_i == t_j) & (d_i | d_j))

@numba.jit(nopython=True)
def _is_comparable_antolini(t_i, t_j, d_i, d_j):
    return ((t_i < t_j) & d_i) | ((t_i == t_j) & d_i & (d_j == 0))

@numba.jit(nopython=True)
def _is_concordant(s_i, s_j, t_i, t_j, d_i, d_j):
    conc = 0.
    if t_i < t_j:
        conc = (s_i < s_j) + (s_i == s_j) * 0.5
    elif t_i == t_j: 
        if d_i & d_j:
            conc = 1. - (s_i != s_j) * 0.5
        elif d_i:
            conc = (s_i < s_j) + (s_i == s_j) * 0.5 
        elif d_j:
            conc = (s_i > s_j) + (s_i == s_j) * 0.5
    return conc * _is_comparable(t_i, t_j, d_i, d_j)

@numba.jit(nopython=True)
def _is_concordant_antolini(s_i, s_j, t_i, t_j, d_i, d_j):
    return (s_i < s_j) & _is_comparable_antolini(t_i, t_j, d_i, d_j)

@numba.jit(nopython=True, parallel=True)
def _sum_comparable(t, d, is_comparable_func):
    n = t.shape[0]
    count = 0.
    for i in numba.prange(n):
        for j in range(n):
            if j != i:
                count += is_comparable_func(t[i], t[j], d[i], d[j])
    return count

@numba.jit(nopython=True, parallel=True)
def _sum_concordant(s, t, d):
    n = len(t)
    count = 0.
    for i in numba.prange(n):
        for j in range(n):
            if j != i:
                count += _is_concordant(s[i, i], s[i, j], t[i], t[j], d[i], d[j])
    return count

@numba.jit(nopython=True, parallel=True)
def _sum_concordant_disc(s, t, d, s_idx, is_concordant_func):
    n = len(t)
    count = 0
    for i in numba.prange(n):
        idx = s_idx[i]
        for j in range(n):
            if j != i:
                count += is_concordant_func(s[idx, i], s[idx, j], t[i], t[j], d[i], d[j])
    return count

def concordance_td(durations, events, surv, surv_idx, method='adj_antolini'):
    """Time dependent concorance index from
    Antolini, L.; Boracchi, P.; and Biganzoli, E. 2005. A timedependent discrimination
    index for survival data. Statistics in Medicine 24:3927–3944.

    If 'method' is 'antolini', the concordance from Antolini et al. is computed.
    
    If 'method' is 'adj_antolini' (default) we have made a small modifications
    for ties in predictions and event times.
    We have followed step 3. in Sec 5.1. in Random Survial Forests paper, except for the last
    point with "T_i = T_j, but not both are deaths", as that doesn't make much sense.
    See '_is_concordant'.

    Arguments:
        durations {np.array[n]} -- Event times (or censoring times.)
        events {np.array[n]} -- Event indicators (0 is censoring).
        surv {np.array[n_times, n]} -- Survival function (each row is a duraratoin, and each col
            is an individual).
        surv_idx {np.array[n_test]} -- Mapping of survival_func s.t. 'surv_idx[i]' gives index in
            'surv' corresponding to the event time of individual 'i'.

    Keyword Arguments:
        method {str} -- Type of c-index 'antolini' or 'adj_antolini' (default {'adj_antolini'}).

    Returns:
        float -- Time dependent concordance index.
    """
    if np.isfortran(surv):
        surv = np.array(surv, order='C')
    assert durations.shape[0] == surv.shape[1] == surv_idx.shape[0] == events.shape[0]
    assert type(durations) is type(events) is type(surv) is type(surv_idx) is np.ndarray
    if events.dtype in ('float', 'float32'):
        events = events.astype('int32')
    if method == 'adj_antolini':
        is_concordant = _is_concordant
        is_comparable = _is_comparable
        return (_sum_concordant_disc(surv, durations, events, surv_idx, is_concordant) /
                _sum_comparable(durations, events, is_comparable))
    elif method == 'antolini':
        is_concordant = _is_concordant_antolini
        is_comparable = _is_comparable_antolini
        return (_sum_concordant_disc(surv, durations, events, surv_idx, is_concordant) /
                _sum_comparable(durations, events, is_comparable))
    return ValueError(f"Need 'method' to be e.g. 'antolini', got '{method}'.")

def _make_surv_idx_from_grid(times_grid, durations):
    times = np.asarray(times_grid, dtype=float)
    t = np.asarray(durations, dtype=float)
    idx = np.searchsorted(times, t, side="right") - 1
    idx[idx < 0] = 0
    last = len(times) - 1
    idx[idx > last] = last
    return idx.astype(np.int64)

def cindex_from_predictions(durations, events, horizons, S_pred, method='antolini'):
    durations = np.asarray(durations, dtype=float)
    events    = np.asarray(events).astype(np.int32)
    horizons  = np.asarray(horizons, dtype=float)
    S_pred    = np.asarray(S_pred, dtype=float)
    assert S_pred.shape[0] == durations.shape[0]
    assert S_pred.shape[1] == horizons.shape[0]
    surv = S_pred.T.copy(order="C")                 # (n_times, n)
    surv_idx = _make_surv_idx_from_grid(horizons, durations)
    
    return float(concordance_td(durations, events, surv, surv_idx, method=method))
def integrated_brier_score(durations_train, events_train,
                           durations_test, events_test,
                           predictions, times_grid):
    """Compute the integrated brier score (IBS) for survival predictions.

    Arguments:
        durations_train {np.array[n_train]} -- Event times (or censoring times) for training data.
        events_train {np.array[n_train]} -- Event indicators (0 is censoring) for training data.
        durations_test {np.array[n_test]} -- Event times (or censoring times) for test data.
        events_test {np.array[n_test]} -- Event indicators (0 is censoring) for test data.
        predictions {np.array[n_test, n_times]} -- Predicted survival function for test data.
        times_grid {np.array[n_times]} -- Time grid corresponding to predictions.

    Returns:
        float -- Integrated Brier Score (IBS).
    """
    durations_train = np.asarray(durations_train, dtype=float)
    events_train = np.asarray(events_train).astype(np.int32)
    durations_test = np.asarray(durations_test, dtype=float)
    events_test = np.asarray(events_test).astype(np.int32)
    predictions = np.asarray(predictions, dtype=float)
    times_grid = np.asarray(times_grid, dtype=float)

    assert predictions.shape[0] == durations_test.shape[0]
    assert predictions.shape[1] == times_grid.shape[0]

    # Compute the Brier score for each time point
    brier_scores = []
    for t in times_grid:
        # Get the predicted survival probabilities at time t
        S_pred_t = predictions[:, np.searchsorted(times_grid, t, side="right") - 1]

        # Compute the Brier score at time t
        brier_score_t = np.mean((S_pred_t - (events_test * (durations_test >= t))) ** 2)
        brier_scores.append(brier_score_t)

    # Compute the integrated Brier Score (IBS)
    ibs = np.trapz(brier_scores, times_grid)

    return ibs