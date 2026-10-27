# -*- coding: utf-8 -*-
"""Week 8 discrepancy experiments on the LQR replication run.

Experiment A (robustness/solver): replace RK4 with the paper's fixed-step
  backward Euler at Ts = 0.01 s and recompute the Table 9 indices.
Experiment B (parameter variation): disable the friction terms (f = 0) and
  recompute; isolates the friction-induced steady-state offsets.
Experiment C (window alignment): recompute our RK4 indices over
  [t_switch, t_switch + 15] s instead of the paper's [5, 20] s, to test the
  hypothesis that the deviation is concentrated in the swing-up phase timing.

Outputs: week8_experiments.csv (+ console table).
"""
import numpy as np
from rip_model import (linearise, lqr_gain, simulate, lqr_controller_factory,
                       indices, PAPER_LQR, Q_LQR, R_LQR, TS, deriv,
                       SWITCH_THRESH, P)

A, B = linearise(with_friction=False)
K = lqr_gain(A, B, Q_LQR, R_LQR)


# ---------------------------------------------------------------- experiment A
def backward_euler_sim(controller, t_end=20.0, n_iter=8):
    """Paper solver: fixed-step backward Euler at Ts = 0.01, ZOH control."""
    from rip_model import Q0_HANGING
    n = int(round(t_end / TS))
    X = np.zeros((n + 1, 4)); U = np.zeros(n + 1); T = np.zeros(n + 1)
    X[0] = [0.0, Q0_HANGING, 0.0, 0.0]
    sw = {'done': False}

    def ctrl(x):
        if not sw['done'] and abs(x[1]) <= SWITCH_THRESH:
            sw['done'] = True
        if sw['done']:
            return float(-(K @ x).item())
        from rip_model import u_swing
        return u_swing(x[1], x[3])

    for k in range(n):
        u = ctrl(X[k]); U[k] = u
        x = X[k].copy()
        for _ in range(n_iter):                     # fixed-point iteration
            x = X[k] + TS * deriv(x, u)
        X[k + 1] = x
        T[k + 1] = (k + 1) * TS
    U[-1] = U[-2]
    return T, X, U


# ---------------------------------------------------------------- experiment B
def frictionless_bias(x):
    """bias() with friction terms zeroed -> used by monkey-patching deriv."""
    q1, q2, qd1, qd2 = x
    s2, c2 = np.sin(q2), np.cos(q2)
    c11 = 0.5 * P['p2'] * np.sin(2 * q2) * qd2
    c12 = -P['p3'] * s2 * qd2 + 0.5 * P['p2'] * np.sin(2 * q2) * qd1
    c21 = -0.5 * P['p2'] * np.sin(2 * q2) * qd1
    g21 = -P['p5'] * s2
    return np.array([c11 * qd1 + c12 * qd2, c21 * qd1 + g21])


def run_with(frictionless=False):
    import rip_model
    orig_bias, orig_deriv = rip_model.bias, rip_model.deriv

    def deriv_nofric(x, u):
        qd = x[2:]
        Minv = np.linalg.inv(rip_model.mass_matrix(x[1]))
        qdd = Minv @ (np.array([u, 0.0]) - frictionless_bias(x))
        return np.concatenate((qd, qdd))

    if frictionless:
        rip_model.deriv = deriv_nofric
    try:
        T, X, U = simulate(lqr_controller_factory(K))
    finally:
        rip_model.deriv, rip_model.bias = orig_deriv, orig_bias
    return T, X, U


# ---------------------------------------------------------------- experiment C
def aligned_indices(T, X, U):
    """Same index code, but window starts at the switch instant."""
    import rip_model
    t_sw = np.where(np.abs(X[:, 1]) <= SWITCH_THRESH)[0][0] * TS
    shift = t_sw - 5.0
    T2 = T - shift                       # paper's [5,20] window -> our [t_sw, t_sw+15]
    return indices(T2, X, U), t_sw


# ---------------------------------------------------------------- run all
rows = []

T, X, U = simulate(lqr_controller_factory(K))
idx_rk4 = indices(T, X, U)
rows.append(('RK4 baseline (friction)', idx_rk4))

T_be, X_be, U_be = backward_euler_sim(None)
idx_be = indices(T_be, X_be, U_be)
rows.append(('Backward Euler (paper solver)', idx_be))

T_nf, X_nf, U_nf = run_with(frictionless=True)
idx_nf = indices(T_nf, X_nf, U_nf)
rows.append(('RK4, friction disabled', idx_nf))

idx_al, t_sw = aligned_indices(T, X, U)
rows.append((f'RK4, window aligned to switch (t={t_sw:.2f}s)', idx_al))

keys = list(PAPER_LQR.keys())
hdr = f"{'variant':44s}" + ''.join(f'{k:>11s}' for k in keys)
print(hdr)
print(f"{'paper Table 9 (LQR)':44s}" + ''.join(f"{PAPER_LQR[k]:11.4f}" for k in keys))
with open('week8_experiments.csv', 'w', encoding='utf-8') as f:
    f.write('variant,' + ','.join(keys) + '\n')
    f.write('paper_LQR,' + ','.join(str(PAPER_LQR[k]) for k in keys) + '\n')
    for name, idx in rows:
        line = f"{name:44s}" + ''.join(f"{idx[k]:11.4f}" for k in keys)
        print(line)
        f.write(name + ',' + ','.join(f'{idx[k]:.4f}' for k in keys) + '\n')
print('\nsaved: week8_experiments.csv')
print('final states: RK4 q2=%.4f q1=%.4f | BE q2=%.4f q1=%.4f | nofric q2=%.4f q1=%.4f'
      % (X[-1, 1], X[-1, 0], X_be[-1, 1], X_be[-1, 0], X_nf[-1, 1], X_nf[-1, 0]))
np.savez('week8_variants.npz', t_rk4=T, X_rk4=X, U_rk4=U,
         t_be=T_be, X_be=X_be, U_be=U_be, t_nf=T_nf, X_nf=X_nf, U_nf=U_nf)
