# -*- coding: utf-8 -*-
"""Week 7 evidence generator: LQR end-to-end replication run.

Outputs (saved next to this file):
  week7_LQR_traj.npz   - trajectory (t, q1, q2, qd1, qd2, u) and switch time
  week7_LQR_fig19.png  - Figure 19-style responses (q1, q2, u vs t)
  week7_LQR_indices.csv- our indices vs the paper's Table 9 LQR column
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rip_model import (linearise, lqr_gain, simulate, lqr_controller_factory,
                       indices, PAPER_LQR, Q_LQR, R_LQR, TS)

A, B = linearise(with_friction=False)
K = lqr_gain(A, B, Q_LQR, R_LQR)
T, X, U = simulate(lqr_controller_factory(K))
sw = np.where(np.abs(X[:, 1]) <= np.pi / 9)[0][0] * TS
idx = indices(T, X, U)

np.savez('week7_LQR_traj.npz', t=T, q1=X[:, 0], q2=X[:, 1],
         qd1=X[:, 2], qd2=X[:, 3], u=U, t_switch=sw, K=K)

# ---------------------------------------------------------------- figure
fig, axes = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
axes[0].plot(T, X[:, 0], lw=1.2, color='#1f77b4')
axes[0].set_ylabel(r'$q_1$ [rad]')
axes[1].plot(T, X[:, 1], lw=1.2, color='#d62728')
axes[1].axhline(np.pi / 9, ls=':', c='gray', lw=0.8)
axes[1].axhline(-np.pi / 9, ls=':', c='gray', lw=0.8)
axes[1].set_ylabel(r'$q_2$ [rad]  (dotted: switch band $\pm\pi/9$)')
axes[2].plot(T, U, lw=0.8, color='#2ca02c')
axes[2].set_ylabel(r'$u$ [N m]')
axes[2].set_xlabel('t [s]')
for ax in axes:
    ax.axvline(5.0, ls='--', c='k', lw=0.7, alpha=0.6)
    ax.grid(alpha=0.3)
axes[0].set_title(f'LQR replication run (switch to LQR at t = {sw:.2f} s)')
fig.tight_layout()
fig.savefig('week7_LQR_fig19.png', dpi=150)

# ---------------------------------------------------------------- table
with open('week7_LQR_indices.csv', 'w', encoding='utf-8') as f:
    f.write('index,ours,paper_Table9_LQR,rel_dev_percent\n')
    for k, v in PAPER_LQR.items():
        rel = 100 * (idx[k] - v) / v if np.isfinite(idx[k]) else float('nan')
        f.write(f'{k},{idx[k]:.4f},{v},{rel:.1f}\n')

print(f'switch to LQR at t = {sw:.2f} s')
print(f"{'index':10s} {'ours':>10s} {'paper':>10s} {'dev%':>7s}")
for k, v in PAPER_LQR.items():
    rel = 100 * (idx[k] - v) / v
    print(f"{k:10s} {idx[k]:10.4f} {v:10.4f} {rel:7.1f}")
print('saved: week7_LQR_traj.npz, week7_LQR_fig19.png, week7_LQR_indices.csv')
