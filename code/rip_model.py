# -*- coding: utf-8 -*-
"""
Rotary Inverted Pendulum (RIP) replication — reference implementation.

Paper: Hernandez, Garcia-Hernandez, Jurado, "Modeling, Simulation, and Control
of a Rotary Inverted Pendulum: A Reinforcement Learning-Based Control
Approach", Modelling 2024, 5, 1824-1852. DOI 10.3390/modelling5040095

Implements:
  - nonlinear dynamics (paper Eq. (2), M/C/g/f elements, Section 3.1)
  - identified parameters p1..p9 (paper Table 5)
  - linearised model at upright (paper Eqs. (7)-(8)) + LQR design
  - energy-based swing-up (paper Eq. (36)-(37), Section 5.7) + switch logic
  - closed-loop simulation + Table 9 performance indices

States: x = [q1, q2, qd1, qd2];  q2 = 0 upright (unstable), q2 = +/-pi hanging.
Angles in rad, torque u in N.m.
"""
import numpy as np
from scipy.linalg import solve_continuous_are as solve_care

# ---------------------------------------------------------------- parameters
# Table 5: identified lumped parameters (least-squares over the Simscape model)
P = dict(
    p1=0.04651,  # J1 + m2 L1^2              [kg m^2]
    p2=0.00448,  # m2 l2^2                   [kg m^2]
    p3=0.01014,  # L1 l2 m2                  [kg m^2]
    p4=0.00446,  # J2 + m2 l2^2              [kg m^2]
    p5=0.22936,  # l2 m2 g                   [kg m^2/s^2]
    p6=0.00953,  # viscous friction, joint 1 [N m s/rad]
    p7=0.00090,  # viscous friction, joint 2 [N m s/rad]
    p8=0.01836,  # Coulomb friction, joint 1 [N m]
    p9=0.00877,  # Coulomb friction, joint 2 [N m]
)
G = 9.81
BETA_C = 100.0      # Coulomb velocity threshold slope: beta_c = 1/w_c, w_c = 0.1/10
J2_SWING = 0.000313  # paper Section 5.7 (J2 = p2/12 as printed)

# Table 8 (MPC) not used here. LQR weights, paper Section 5.1:
Q_LQR = np.diag([1.0, 1.0, 1.0, 1.0])
R_LQR = 1.0

# Swing-up, Section 5.7:
K_SW, N_SW = 0.5, 0.05
SWITCH_THRESH = np.pi / 9          # |q2| <= 20 deg -> stabilising controller
Q0_HANGING = 17 * np.pi / 18       # initial pendulum angle (scenario 1)

TS = 0.01          # sampling interval (paper: fixed-step, 0.01 s)
T_END = 20.0


# ---------------------------------------------------------------- dynamics
def mass_matrix(q2):
    return np.array([[P['p1'] + P['p2'] * np.sin(q2) ** 2, P['p3'] * np.cos(q2)],
                     [P['p3'] * np.cos(q2), P['p4']]])


def bias(q):
    """C(q,qd)qd + g(q) + f(qd)   (paper Section 3.1 element definitions)."""
    q1, q2, qd1, qd2 = q
    s2, c2 = np.sin(q2), np.cos(q2)
    c11 = 0.5 * P['p2'] * np.sin(2 * q2) * qd2
    c12 = -P['p3'] * s2 * qd2 + 0.5 * P['p2'] * np.sin(2 * q2) * qd1
    c21 = -0.5 * P['p2'] * np.sin(2 * q2) * qd1
    g21 = -P['p5'] * s2
    f1 = P['p8'] * np.tanh(BETA_C * qd1) + P['p6'] * qd1
    f2 = P['p9'] * np.tanh(BETA_C * qd2) + P['p7'] * qd2
    return np.array([c11 * qd1 + c12 * qd2 + f1,
                     c21 * qd1 + g21 + f2])


def deriv(x, u):
    qd = x[2:]
    Minv = np.linalg.inv(mass_matrix(x[1]))
    qdd = Minv @ (np.array([u, 0.0]) - bias(x))
    return np.concatenate((qd, qdd))


# ---------------------------------------------------------------- linearisation + LQR
def linearise(with_friction=True):
    """Linearise Eq. (2) about x = 0 (upright). Friction is linearised as
    tanh(beta*qd) ~ beta*qd (consistent with the paper's continuous model)."""
    M0 = np.array([[P['p1'], P['p3']], [P['p3'], P['p4']]])
    Minv = np.linalg.inv(M0)
    D = P['p1'] * P['p4'] - P['p3'] ** 2
    A = np.zeros((4, 4))
    B = np.zeros((4, 1))
    A[0, 2] = A[1, 3] = 1.0
    A[2, 1] = -P['p3'] * P['p5'] / D
    A[3, 1] = P['p1'] * P['p5'] / D
    B[2, 0] = P['p4'] / D
    B[3, 0] = -P['p3'] / D
    if with_friction:
        d1 = P['p6'] + BETA_C * P['p8']
        d2 = P['p7'] + BETA_C * P['p9']
        A[2, 2] -= Minv[0, 0] * d1
        A[2, 3] -= Minv[0, 1] * d2
        A[3, 2] -= Minv[1, 0] * d1
        A[3, 3] -= Minv[1, 1] * d2
    return A, B


def lqr_gain(A, B, Q, R):
    S = solve_care(A, B, Q, np.array([[R]]))
    return (np.array([[R]]) ** -1 @ B.T @ S)


# ---------------------------------------------------------------- swing-up
def swing_up_energy(q2, qd2, literal_paper=False):
    """Paper Eq. (37): E = 1/2 J2 qd2^2 + p_i (cos q2 - 1);  E0 = 0 upright.
    The printed equation uses p2 (=0.00448), which yields a pumping torque two
    orders of magnitude too small and no swing-up occurs (verified
    experimentally).  The dimensionally consistent coefficient is p5 (=m2 l2 g),
    which reproduces the behaviour of Figure 19; it is the default and this
    interpretation is recorded as a documented deviation."""
    coef = P['p2'] if literal_paper else P['p5']
    return 0.5 * J2_SWING * qd2 ** 2 + coef * (np.cos(q2) - 1.0)


def u_swing(q2, qd2):
    """Paper Eq. (36): usw = n g tanh(k (E - E0) qd2 cos q2)."""
    E = swing_up_energy(q2, qd2)
    return N_SW * G * np.tanh(K_SW * (E - 0.0) * qd2 * np.cos(q2))


def u_swing_paper(q2, qd2):
    """Literal Eq. (36)-(37) with the printed p2 coefficient (kept for the
    deviation study; produces no swing-up -- see swing_up_energy docstring)."""
    E = swing_up_energy(q2, qd2, literal_paper=True)
    return N_SW * G * np.tanh(K_SW * (E - 0.0) * qd2 * np.cos(q2))


# ---------------------------------------------------------------- simulation
def rk4_step(x, u, h):
    k1 = deriv(x, u)
    k2 = deriv(x + 0.5 * h * k1, u)
    k3 = deriv(x + 0.5 * h * k2, u)
    k4 = deriv(x + h * k3, u)
    return x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate(controller, x0=None, t_end=T_END, n_sub=10, disturbance=None):
    """Fixed controller sampling TS (ZOH); RK4 integration with n_sub substeps.
    controller(x, t) -> u. Returns t, X (N x 4), U (N,)."""
    x0 = np.array([0.0, Q0_HANGING, 0.0, 0.0]) if x0 is None else np.asarray(x0, float)
    n = int(round(t_end / TS))
    X = np.zeros((n + 1, 4)); U = np.zeros(n + 1); T = np.zeros(n + 1)
    X[0] = x0
    h = TS / n_sub
    for k in range(n):
        t = k * TS
        u = controller(X[k], t)
        if disturbance is not None:
            u = u + disturbance(t)
        U[k] = u
        x = X[k].copy()
        for _ in range(n_sub):
            x = rk4_step(x, u, h)
        X[k + 1] = x
        T[k + 1] = (k + 1) * TS
    U[-1] = U[-2]
    return T, X, U


def lqr_controller_factory(K):
    switched = {'done': False}

    def ctrl(x, t):
        q1, q2 = x[0], x[1]
        if not switched['done'] and abs(q2) <= SWITCH_THRESH:
            switched['done'] = True
        if switched['done']:
            return float(-(K @ x).item())
        return float(u_swing(q2, x[3]))
    return ctrl


# ---------------------------------------------------------------- metrics
def indices(T, X, U):
    """Table 9 indices. e1 = qd1 - q1 = -q1 (qd = 0), e2 = -q2.
    RMS{e} is computed on the stacked error vector [e1; e2] (definition TBC
    against the paper -- documented open item)."""
    e1, e2 = -X[:, 0], -X[:, 1]
    m1 = (T > 5) & (T <= 10)
    m2 = (T >= 5) & (T <= 20)
    out = dict(
        max_e1=np.max(np.abs(e1[m1])),
        rms_e1=np.sqrt(np.mean(e1[m2] ** 2)),
        max_e2=np.max(np.abs(e2[m1])),
        rms_e2=np.sqrt(np.mean(e2[m2] ** 2)),
        rms_e=np.sqrt(np.mean(e1[m2] ** 2 + e2[m2] ** 2)),
        rms_u=np.sqrt(np.mean(U[m2] ** 2)),
    )
    # settling time: paper defines it as q2 remaining within the limits of its
    # steady-state upper/lower values.  Implemented as: the first time after
    # which q2 stays within +/-5% of the max excursion about its final value.
    q2f = X[-1, 1]
    band = 0.05 * np.max(np.abs(X[m2, 1]))
    outside = np.where(np.abs(X[:, 1] - q2f) > band)[0]
    outside = outside[outside >= int(5 / TS)] if len(outside) else outside
    if len(outside):
        t_settle = TS * (outside[-1] + 1)
    else:
        t_settle = 5.0
    out['settling'] = min(t_settle, 20.0)
    # overshoot: max |q2| excursion in [5, 20] after first zero crossing
    zc = np.where(np.sign(X[:-1, 1]) * np.sign(X[1:, 1]) < 0)[0]
    zc = zc[zc >= int(5 / TS)]
    out['overshoot'] = np.max(np.abs(e2[m2])) if len(zc) == 0 else np.max(
        np.abs(X[zc[0]:, 1]))
    return out


PAPER_LQR = dict(max_e1=0.7422, rms_e1=0.2099, max_e2=0.2205, rms_e2=0.0401,
                 rms_e=0.2099, rms_u=0.0573, overshoot=0.0609, settling=8.63)


if __name__ == '__main__':
    # LQR designed on the frictionless linearisation (validated: reproduces
    # the paper's printed gain vector exactly -- see run_week7.py).
    A, B = linearise(with_friction=False)
    K = lqr_gain(A, B, Q_LQR, R_LQR)
    print('A =\n', np.round(A, 4))
    print('B =', np.round(B.ravel(), 4))
    K_paper_order = [K[0, 0], K[0, 2], K[0, 1], K[0, 3]]  # [q1 qd1 q2 qd2]
    print('LQR gain K ours  [q1 qd1 q2 qd2] =', np.round(K_paper_order, 4))
    print('LQR gain K paper [q1 qd1 q2 qd2] = [-1.0000 -1.3805 -11.2936 -1.7981]')
    T, X, U = simulate(lqr_controller_factory(K))
    print('final |q2| =', abs(X[-1, 1]), ' final |q1| =', abs(X[-1, 0]))
    idx = indices(T, X, U)
    print(f"{'index':10s} {'ours':>10s} {'paper':>10s}")
    for k in PAPER_LQR:
        print(f"{k:10s} {idx[k]:10.4f} {PAPER_LQR[k]:10.4f}")
    np.save('week7_LQR_traj.npy', np.column_stack([T, X, U]))
