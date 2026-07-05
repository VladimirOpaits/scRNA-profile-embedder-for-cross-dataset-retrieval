import numpy as np


def cosine_distance_matrix(M):
    M = M / np.linalg.norm(M, axis=1, keepdims=True)
    return 1.0 - M @ M.T


def greedy_farthest(D, cohort, pool, k):
    cohort = list(cohort)
    pool = list(pool)
    order = []
    for _ in range(min(k, len(pool))):
        d = D[np.ix_(pool, cohort)].min(axis=1)
        best = int(np.argmax(d))
        chosen = pool[best]
        order.append(chosen)
        cohort.append(chosen)
        pool.pop(best)
    return order


def greedy_quantile(D, cohort, pool, k, q):
    cohort = list(cohort)
    pool = list(pool)
    order = []
    for _ in range(min(k, len(pool))):
        d = D[np.ix_(pool, cohort)].min(axis=1)
        target = np.quantile(d, q)
        best = int(np.argmin(np.abs(d - target)))
        chosen = pool[best]
        order.append(chosen)
        cohort.append(chosen)
        pool.pop(best)
    return order
