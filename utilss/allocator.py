import numpy as np

def greedy_swap_allocator(phi_matrix):
    """
    Greedy + one-swap heuristic for channel assignment.

    Args
      phi_matrix :  (N, M)  semantic spectral efficiency Φ for each user/channel

    Returns
      user2chan  :  list of length N  (channel index or –1 if none)
    """
    N, M = phi_matrix.shape
    assigned = [-1] * N         # final mapping
    ch_used  = [False] * M

    # 1) Greedy – largest Φ first
    flat = [(-phi_matrix[u, m], u, m) for u in range(N) for m in range(M)]
    flat.sort()                 # ascending because of –Φ

    for _, u, m in flat:
        if assigned[u] == -1 and not ch_used[m]:
            assigned[u] = m
            ch_used[m] = True

    # 2) Local 1-swap improvement
    improved = True
    while improved:
        improved = False
        for u in range(N):
            for v in range(u + 1, N):
                mu, mv = assigned[u], assigned[v]
                if mu == -1 or mv == -1:           # skip unassigned
                    continue
                # check if swapping channels helps both
                before = phi_matrix[u, mu] + phi_matrix[v, mv]
                after  = phi_matrix[u, mv] + phi_matrix[v, mu]
                if after > before:
                    assigned[u], assigned[v] = mv, mu
                    improved = True
    return assigned
