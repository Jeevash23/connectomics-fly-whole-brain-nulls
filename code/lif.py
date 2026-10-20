"""Shiu et al. (Nature 2024) leaky integrate-and-fire whole-brain model, batched over trials with numba.

Follows Shiu et al.'s model.py (data/shiu2024-repo, commit 91bdd1e) as run by Brian2 with its default 0.1 ms step:

    dv/dt = (v_0 - v + g) / t_mbr   dg/dt = -g / tau   (unless refractory), method='linear' (exact solution)
    v_0 = v_rst = -52 mV, v_th = -45 mV, t_mbr = 20 ms, tau = 5 ms, t_rfc = 2.2 ms, t_dly = 1.8 ms, w_syn = 0.275 mV
    synapses: on_pre g += w with w = (sign x synapses) x w_syn, delay 1.8 ms (18 steps)
    Poisson input: PoissonInput(target_var='v', N=1, weight = 250 x w_syn); Poisson targets have refractory period 0
    silencing: all outgoing weights of a silenced neuron set to 0

Brian2's schedule each step: state update -> thresholder -> synapses (delayed spikes and Poisson input) -> resetter.
The thresholder sets not_refractory = (steps since last spike >= refractory steps) and then tests v > v_th and
not_refractory; a spike sets not_refractory = False. The state updater runs before the thresholder, so it uses the
previous step's not_refractory: a neuron with a 2.2 ms period is not integrated for the 22 steps after its spike and can
spike again from the 22nd; a Poisson target (period 0) skips integration for 1 step after each spike.

Two implementations: _run_dense updates every neuron each step (reference); _run_active skips neurons at exact rest,
which do not change without input. Both give identical spike counts for the same seed (scripts/check_dense_vs_active.py).
A run returns spike counts per trial and neuron; over 1,000 ms the count is the firing rate in Hz.
"""
import math
import numpy as np
from numba import njit, prange

V_REST = -52.0
V_RESET = -52.0
V_TH = -45.0
TAU_M = 20.0
TAU_S = 5.0
DT = 0.1
DELAY = 18
REFRACTORY = 22
W_SYN = 0.275
POISSON_SCALE = 250.0
E_M = math.exp(-DT / TAU_M)
E_S = math.exp(-DT / TAU_S)
C_G = TAU_S / (TAU_S - TAU_M) * (E_S - E_M)   # exact contribution of g to v over one step
REST_EPS = 1e-9


@njit(parallel=True, cache=True)
def _run_dense(indptr, indices, weights, n, drive_idx, drive_rate, silenced, n_trials, n_steps, seed):
    counts = np.zeros((n_trials, n), dtype=np.int32)
    kick = POISSON_SCALE * W_SYN
    slots = DELAY + 1
    rfc = np.full(n, REFRACTORY, dtype=np.int64)
    for k in range(drive_idx.size):
        rfc[drive_idx[k]] = 0
    for tr in prange(n_trials):
        np.random.seed(seed + tr)
        v = np.full(n, V_REST, dtype=np.float64)
        g = np.zeros(n, dtype=np.float64)
        free = np.ones(n, dtype=np.bool_)          # not_refractory as left by the last thresholder
        last = np.full(n, -10**9, dtype=np.int64)
        queue = np.zeros((slots, n), dtype=np.float64)
        spiked = np.empty(n, dtype=np.int32)
        for s in range(n_steps):
            for i in range(n):                      # state update
                if free[i]:
                    vi = v[i]
                    gi = g[i]
                    v[i] = V_REST + (vi - V_REST) * E_M + gi * C_G
                    g[i] = gi * E_S
            ns = 0
            for i in range(n):                      # thresholder
                f = s - last[i] >= rfc[i]
                free[i] = f
                if f and v[i] > V_TH:
                    spiked[ns] = i
                    ns += 1
                    free[i] = False
                    last[i] = s
            slot = s % slots                         # synapses
            for i in range(n):
                if queue[slot, i] != 0.0:
                    g[i] += queue[slot, i]
                    queue[slot, i] = 0.0
            later = (s + DELAY) % slots
            for k in range(ns):
                pre = spiked[k]
                counts[tr, pre] += 1
                if silenced[pre]:
                    continue
                for e in range(indptr[pre], indptr[pre + 1]):
                    queue[later, indices[e]] += weights[e]
            for k in range(drive_idx.size):
                if np.random.random() < drive_rate[k] * DT * 1e-3:
                    v[drive_idx[k]] += kick
            for k in range(ns):                     # resetter
                i = spiked[k]
                v[i] = V_RESET
                g[i] = 0.0
    return counts


@njit(parallel=True, cache=True)
def _run_active(indptr, indices, weights, n, drive_idx, drive_rate, silenced, n_trials, n_steps, seed):
    counts = np.zeros((n_trials, n), dtype=np.int32)
    kick = POISSON_SCALE * W_SYN
    slots = DELAY + 1
    rfc = np.full(n, REFRACTORY, dtype=np.int64)
    is_driven = np.zeros(n, dtype=np.bool_)
    for k in range(drive_idx.size):
        rfc[drive_idx[k]] = 0
        is_driven[drive_idx[k]] = True
    for tr in prange(n_trials):
        np.random.seed(seed + tr)
        v = np.full(n, V_REST, dtype=np.float64)
        g = np.zeros(n, dtype=np.float64)
        free = np.ones(n, dtype=np.bool_)
        last = np.full(n, -10**9, dtype=np.int64)
        queue = np.zeros((slots, n), dtype=np.float64)
        touched = np.zeros((slots, n), dtype=np.int32)
        n_touched = np.zeros(slots, dtype=np.int64)
        marked = np.zeros((slots, n), dtype=np.bool_)
        active = np.empty(n, dtype=np.int32)
        nxt = np.empty(n, dtype=np.int32)
        in_active = np.zeros(n, dtype=np.bool_)
        n_active = 0
        for k in range(drive_idx.size):
            i = drive_idx[k]
            if not in_active[i]:
                in_active[i] = True
                active[n_active] = i
                n_active += 1
        spiked = np.empty(n, dtype=np.int32)
        for s in range(n_steps):
            for a in range(n_active):               # state update
                i = active[a]
                if free[i]:
                    vi = v[i]
                    gi = g[i]
                    v[i] = V_REST + (vi - V_REST) * E_M + gi * C_G
                    g[i] = gi * E_S
            ns = 0
            for a in range(n_active):               # thresholder (inactive neurons are free and at rest)
                i = active[a]
                f = s - last[i] >= rfc[i]
                free[i] = f
                if f and v[i] > V_TH:
                    spiked[ns] = i
                    ns += 1
                    free[i] = False
                    last[i] = s
            slot = s % slots                         # synapses
            for t in range(n_touched[slot]):
                i = touched[slot, t]
                g[i] += queue[slot, i]
                queue[slot, i] = 0.0
                marked[slot, i] = False
                if not in_active[i]:
                    in_active[i] = True
                    active[n_active] = i
                    n_active += 1
            n_touched[slot] = 0
            later = (s + DELAY) % slots
            for k in range(ns):
                pre = spiked[k]
                counts[tr, pre] += 1
                if silenced[pre]:
                    continue
                for e in range(indptr[pre], indptr[pre + 1]):
                    j = indices[e]
                    queue[later, j] += weights[e]
                    if not marked[later, j]:
                        marked[later, j] = True
                        touched[later, n_touched[later]] = j
                        n_touched[later] += 1
            for k in range(drive_idx.size):
                if np.random.random() < drive_rate[k] * DT * 1e-3:
                    v[drive_idx[k]] += kick
            for k in range(ns):                     # resetter
                i = spiked[k]
                v[i] = V_RESET
                g[i] = 0.0
            m = 0                                    # keep driven, refractory or non-resting neurons
            for a in range(n_active):
                i = active[a]
                if is_driven[i] or not free[i] or abs(v[i] - V_REST) >= REST_EPS or abs(g[i]) >= REST_EPS:
                    nxt[m] = i
                    m += 1
                else:
                    v[i] = V_REST
                    g[i] = 0.0
                    in_active[i] = False
            for a in range(m):
                active[a] = nxt[a]
            n_active = m
    return counts


class Network:
    """Outgoing-edge (CSR) form of a wiring: pre -> post with weight (sign x synapses) x w_syn."""

    def __init__(self, n, pre, post, signed_synapses):
        pre = np.asarray(pre, dtype=np.int64)
        post = np.asarray(post, dtype=np.int64)
        w = np.asarray(signed_synapses, dtype=np.float64)
        order = np.lexsort((post, pre))
        self.n = int(n)
        self.pre, self.post, self.signed = pre[order], post[order], w[order]
        counts = np.bincount(self.pre, minlength=self.n)
        self.indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        self.weights = self.signed * W_SYN

    def run(self, drive, silence=(), n_trials=30, duration_ms=1000.0, seed=0, dense=False):
        """drive: {neuron index: Poisson rate in Hz}; silence: neuron indices whose outputs are removed."""
        idx = np.array(sorted(drive), dtype=np.int64)
        rate = np.array([float(drive[i]) for i in idx], dtype=np.float64)
        silenced = np.zeros(self.n, dtype=np.bool_)
        if len(silence):
            silenced[np.asarray(list(silence), dtype=np.int64)] = True
        n_steps = int(round(duration_ms / DT))
        fn = _run_dense if dense else _run_active
        return fn(self.indptr, self.post, self.weights, self.n, idx, rate, silenced, int(n_trials), n_steps, int(seed))
