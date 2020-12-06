import numpy as np
import pandas as pd
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
            conc = (s_i < s_j) + (s_i == s_j) * 0.5  # different from RSF paper.
        elif d_j:
            conc = (s_i > s_j) + (s_i == s_j) * 0.5  # different from RSF paper.
    return conc * _is_comparable(t_i, t_j, d_i, d_j)

@numba.jit(nopython=True)
def _is_concordant_modified(s_i, s_j, t_i, t_j, d_i, d_j):
    conc = 0.
    if t_i < t_j:
        conc = (s_i > s_j) + (s_i == s_j) * 0.5
    elif t_i == t_j:
        if d_i & d_j:
            conc = 1. - (s_i != s_j) * 0.5
        elif d_i:
            conc = (s_i > s_j) + (s_i == s_j) * 0.5  # different from RSF paper.
        elif d_j:
            conc = (s_i < s_j) + (s_i == s_j) * 0.5  # different from RSF paper.
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
def _sum_comparable_modified(t_g1, t_g2, d_g1, d_g2, is_comparable_func):
    n_g1 = t_g1.shape[0]
    n_g2 = t_g2.shape[0]
    count = 0.
    for i in numba.prange(n_g1):
        for j in range(n_g2):
            count += is_comparable_func(t_g1[i], t_g2[j], d_g1[i], d_g2[j])
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

@numba.jit(nopython=True, parallel=True)
def _sum_concordant_disc_modified(s_g1, s_g2, t_g1, t_g2, d_g1, d_g2, s_idx_g1, s_idx_g2, is_concordant_func):
    n_g1 = len(t_g1)
    n_g2 = len(t_g2)
    count = 0
    for i in numba.prange(n_g1):
        idx_g1 = s_idx_g1[i]
        for j in range(n_g2):
            idx_g2 = s_idx_g2[j]
            count += is_concordant_func(s_g1[idx_g1, i], s_g2[idx_g2, j], t_g1[i], t_g2[j], d_g1[i], d_g2[j])
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
        surv {np.array[n_times, n]} -- Survival function (each row is a duration, and each col
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


def concordance_td_modified_for_groups(durations_g1, events_g1, surv_g1, surv_idx_g1, durations_g2, events_g2, surv_g2, surv_idx_g2, method):
    """
        Returns:
            float -- Modified time dependent concordance index.
    """
    # Group 1
    assert durations_g1.shape[0] == surv_g1.shape[1] == surv_idx_g1.shape[0] == events_g1.shape[0]
    assert type(durations_g1) is type(events_g1) is type(surv_g1) is type(surv_idx_g1) is np.ndarray
    # Group 2
    assert durations_g2.shape[0] == surv_g2.shape[1] == surv_idx_g2.shape[0] == events_g2.shape[0]
    assert type(durations_g2) is type(events_g2) is type(surv_g2) is type(surv_idx_g2) is np.ndarray

    if events_g1.dtype in ('float', 'float32'):
        events_g1 = events_g1.astype('int32')
    if events_g2.dtype in ('float', 'float32'):
        events_g2 = events_g2.astype('int32')

    if method == 'adj_antolini':
        is_concordant = _is_concordant
        is_comparable = _is_comparable
        return (_sum_concordant_disc_modified(surv_g1, surv_g2, durations_g1, durations_g2, events_g1, events_g2, surv_idx_g1, surv_idx_g2, is_concordant) /
                _sum_comparable_modified(durations_g1, durations_g2, events_g1, events_g2, is_comparable))
    return ValueError(f"Need 'method' to be e.g. 'antolini', got '{method}'.")


def concordance_td_modified_for_groups_time_event(durations_g1, events_g1, surv_g1, surv_idx_g1, durations_g2, events_g2, surv_g2, surv_idx_g2, method):
    """
        Returns:
            float -- Modified time dependent concordance index.
    """
    # Group 1
    assert durations_g1.shape[0] == surv_g1.shape[1] == surv_idx_g1.shape[0] == events_g1.shape[0]
    assert type(durations_g1) is type(events_g1) is type(surv_g1) is type(surv_idx_g1) is np.ndarray
    # Group 2
    assert durations_g2.shape[0] == surv_g2.shape[1] == surv_idx_g2.shape[0] == events_g2.shape[0]
    assert type(durations_g2) is type(events_g2) is type(surv_g2) is type(surv_idx_g2) is np.ndarray

    if events_g1.dtype in ('float', 'float32'):
        events_g1 = events_g1.astype('int32')
    if events_g2.dtype in ('float', 'float32'):
        events_g2 = events_g2.astype('int32')

    if method == 'adj_antolini':
        is_concordant_modified = _is_concordant_modified
        is_comparable_modified = _is_comparable
        return (_sum_concordant_disc_modified(surv_g1, surv_g2, durations_g1, durations_g2, events_g1, events_g2, surv_idx_g1, surv_idx_g2, is_concordant_modified) /
                _sum_comparable_modified(durations_g1, durations_g2, events_g1, events_g2, is_comparable_modified))
    return ValueError(f"Need 'method' to be e.g. 'antolini', got '{method}'.")