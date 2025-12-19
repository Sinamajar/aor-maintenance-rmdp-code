# Sina Shahri Majarshin 
# PhD Research Problem 1
# Experiment 1: Mixture Data Generation and Policy Evaluation

import numpy as np
import pandas as pd
import math
from scipy.stats import chi2
import matplotlib.pyplot as plt

##############################
# Global problem parameters
##############################

St = np.array([0, 1, 2, 3, 4, 5])   # degradation states
K  = len(St)
R  = 22                             # replacement cost
T  = 30                             # horizon (months)
gamma = 1                           # no discounting

def OC(x):
    return (36/125) * x**3         # c(0)=0, c(5)=36

########################################
# Transition matrices for each component
########################################

P_fragile = np.array([
    [0.0500, 0.0750, 0.1000, 0.1000, 0.2250, 0.4500],
    [0.0000, 0.1000, 0.1000, 0.1000, 0.2000, 0.5000],
    [0.0000, 0.0000, 0.1250, 0.1250, 0.2500, 0.5000],
    [0.0000, 0.0000, 0.0000, 0.1667, 0.1667, 0.6667],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.2500, 0.7500],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000]
])

P_moderate = np.array([
    [0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667],
    [0.0000, 0.2000, 0.2000, 0.2000, 0.2000, 0.2000],
    [0.0000, 0.0000, 0.2500, 0.2500, 0.2500, 0.2500],
    [0.0000, 0.0000, 0.0000, 0.3333, 0.3333, 0.3333],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.5000, 0.5000],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000]
])

P_sturdy = np.array([
    [0.5500, 0.1500, 0.1000, 0.1000, 0.0500, 0.0500],
    [0.0000, 0.4500, 0.2500, 0.1500, 0.1000, 0.0500],
    [0.0000, 0.0000, 0.4500, 0.2000, 0.2000, 0.1500],
    [0.0000, 0.0000, 0.0000, 0.4000, 0.3000, 0.3000],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.4000, 0.6000],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000]
])

COMPONENTS = {
    'fragile':  P_fragile,
    'moderate': P_moderate,
    'sturdy':   P_sturdy
}

########################################
# Utility functions
########################################

def normalize_row(p):
    p = np.asarray(p, dtype=float)
    s = p.sum()
    if s <= 0:
        return np.ones_like(p) / len(p)
    return p / s

def compute_hitting_time(P, target_state):
    N = P.shape[0]
    H = np.eye(N); b = np.ones(N); b[target_state] = 0
    for i in range(N):
        if i != target_state:
            H[i, :] -= P[i, :]
    return np.linalg.solve(H, b)[0]

def count_upper_triangular_zeros(matrix):
    upper_triangular_zeros = 0
    for i in range(matrix.shape[0]):
        for j in range(i, matrix.shape[1]):
            if matrix[i, j] == 0:
                upper_triangular_zeros += 1
    return upper_triangular_zeros - 1

########################################
# Training data generation
########################################

def simulate_and_count_transitions(P_train, num_steps, seed=None):
    """Single-matrix generator (used in unbiased case).
       Operator replaces ONLY on failure: row K-1 sends to 0 next period."""
    if seed is not None:
        np.random.seed(seed)

    Nm = P_train.shape[0]
    P_oper = P_train.copy().astype(float)
    P_oper[-1, :] = 0.0; P_oper[-1, 0] = 1.0

    states = np.zeros(num_steps, dtype=int)
    s = 0
    for t in range(num_steps):
        probs = normalize_row(P_oper[s])
        ns = np.random.choice(Nm, p=probs)
        states[t] = s
        s = ns

    TN = np.zeros((Nm, Nm), dtype=int)
    for k in range(len(states) - 1):
        i = states[k]; j = states[k+1]
        TN[i, j] += 1
    return states, TN

def simulate_and_count_transitions_mixture_episodic(P_list, weights=None,
                                                    num_steps=3000, seed=None):
    """
    Episodic mixture generator (used in biased case, per your spec):
      - Start in state 0.
      - At the beginning of each EPISODE, pick ONE component matrix ~ Categorical(weights).
      - Evolve using that SAME matrix until failure (state K-1) is ENTERED.
      - Upon entering failure, record that transition, then 'replace':
          reset state to 0 and pick a NEW component for the next episode.
      - Repeat until the number of RECORDED TRANSITIONS reaches num_steps.

    Returns:
      TN     : (K x K) transition count matrix
      P_bar  : weighted arithmetic mean of P_list (for reporting only)
      weights: normalized weights actually used
    """
    if seed is not None:
        np.random.seed(seed)

    if weights is None:
        weights = np.ones(len(P_list)) / len(P_list)
    weights = np.asarray(weights, dtype=float)
    weights /= weights.sum()

    Nm = P_list[0].shape[0]
    TN = np.zeros((Nm, Nm), dtype=int)

    # Start first episode
    s = 0
    comp_idx = np.random.choice(len(P_list), p=weights)
    transitions = 0

    while transitions < num_steps:
        P_use = P_list[comp_idx]
        probs = normalize_row(P_use[s])
        ns = np.random.choice(Nm, p=probs)

        # Record transition
        TN[s, ns] += 1
        transitions += 1

        if ns == Nm - 1:
            # Failure entered – end episode, replace, pick new component
            s = 0
            comp_idx = np.random.choice(len(P_list), p=weights)
        else:
            s = ns

    # Simple mean matrix for reporting (not used for generation)
    P_bar = np.tensordot(weights, np.stack(P_list, axis=0), axes=(0, 0))
    P_bar[-1, :] = 0.0; P_bar[-1, -1] = 1.0
    return TN, P_bar, weights

def q_apx_factory(TN):
    def q_apx(i):
        row = TN[i, :]
        tot = row.sum()
        if tot == 0:
            feasible = np.arange(i, K)
            tmp = np.zeros(K); tmp[feasible] = 1.0 / len(feasible)
            return tmp
        if i <= (K-2):
            return row / tot
        else:
            Row = row / tot
            return Row[::-1]
    return q_apx

########################################
# Policy evaluation on the TRUE system
########################################

def simulate_trajectory(chosen_actions_matrix, P_true):
    total_cost = 0.0
    s = 0
    for t in range(T):
        a = chosen_actions_matrix[s, t]
        total_cost += OC(s)
        if a == 1:
            total_cost += R
            probs = normalize_row(P_true[0])
            s = np.random.choice(K, p=probs)
        else:
            probs = normalize_row(P_true[s])
            s = np.random.choice(K, p=probs)
    total_cost += OC(s)
    return total_cost

########################################
# Main experiment runner
########################################

def run_experiment(component_type='fragile',
                   data_case='unbiased',
                   training_path_length=2000,
                   seed=123,
                   n_trajectories_eval=10000,
                   mixture_weights=None):
    """
    If data_case == 'unbiased':
        - training data generated from the SAME component as evaluation.
        - generator uses 'replace on failure next period' convention.
    If data_case == 'biased':
        - training data generated from an EPISODIC mixture of {fragile, moderate, sturdy}
          with equal weights (or 'mixture_weights' if provided):
            pick one matrix for a whole component lifetime, run until failure is entered,
            then reset to 0 and pick a new matrix, and so on.
        - We collect exactly 'training_path_length' transitions total.
    """

    # TRUE matrix used for evaluation:
    P_true = COMPONENTS[component_type].copy().astype(float)

    # ---- Generate training data / TN
    if data_case == 'unbiased':
        P_train_source = P_true.copy()
        _, TN = simulate_and_count_transitions(P_train_source, num_steps=training_path_length, seed=seed)
        P_bar = None; wts = None; mixture_flag = False
        src_label = 'self'
    else:
        # EPISODIC mixture of all three components
        P_list = [P_fragile, P_moderate, P_sturdy]
        TN, P_bar, wts = simulate_and_count_transitions_mixture_episodic(
            P_list, weights=mixture_weights, num_steps=training_path_length, seed=seed
        )
        P_train_source = None               # not a single source
        mixture_flag = True
        src_label = 'episodic_mixture(fr,mod,stu)'

    q_apx = q_apx_factory(TN)

    a_values = np.linspace(1e-12, 0.99999999999, 10)

    alpha_values          = []
    avg_costs_empirical   = []
    avg_costs_KL          = []
    avg_costs_MLE         = []
    MLE_stdev_list        = []
    KL_stdev_list         = []
    Empirical_stdev_list  = []

    P_empirical_final     = None

    for alpha in a_values:

        # ---------- KL-based robust DP ----------
        def Beta_Generator(i):
            if i < (K-1):
                I  = K - i
                ni = TN[i, :].sum()
                if ni == 0:
                    return 0.0
                comb_val = math.comb(int(ni + I - 1), I - 1)
                inner = 1 - (1 - alpha)**(1 / max(St))
                if inner <= 0 or comb_val <= 0:
                    return 0.0
                return (1/ni) * np.log(comb_val / inner)
            else:
                return 0.0

        def f(mu, B, qvec, Vvec):
            denom = [mu - Vvec[j] for j in range(K)]
            denom = [d if d > 1e-12 else 1e-12 for d in denom]
            Lambda_opt = (sum(qvec[j] / denom[j] for j in range(K)))**(-1)
            inside_logs = np.maximum(np.array([mu - Vvec[j] for j in range(K)]), 1e-12)
            expQ = np.dot(qvec, np.log(inside_logs))
            return ((B - 1) * Lambda_opt + mu - Lambda_opt * expQ + Lambda_opt * np.log(Lambda_opt))

        def P_solver_KL(i, Vvec):
            phat = q_apx(i)
            B    = Beta_Generator(i)
            Vbar = np.dot(phat, Vvec)
            if B == 0:
                MU = max(Vvec)
            else:
                num = (max(Vvec) - np.exp(-B) * Vbar)
                den = (1 - np.exp(-B)); den = max(den, 1e-16)
                MU = num / den

            def line_minimizer(Vcand):
                eps = 1e-12 / 4
                a = max(Vcand) + eps
                b = MU
                gap = 1e-12 / 4
                while b - a > gap:
                    x1 = a + (b-a)/3
                    x2 = b - (b-a)/3
                    if f(x1, B, phat, Vcand) <= f(x2, B, phat, Vcand):
                        b = x2
                    else:
                        a = x1
                return f(a, B, phat, Vcand)

            return line_minimizer(Vvec)

        memoKL = {}
        chosen_actions_matrix_KL = np.zeros((K, T)) + 999

        def W_KL(n, i):
            if n == T: return OC(i)
            if (n, i) in memoKL: return memoKL[(n, i)]
            future_vals = np.array([W_KL(n+1, s) for s in range(K)])
            repl_cost   = R + gamma * P_solver_KL(0, future_vals)
            norepl_cost = gamma * P_solver_KL(i, future_vals)
            act = 1 if repl_cost <= norepl_cost else 0
            chosen_actions_matrix_KL[i, n] = act
            val = OC(i) + min(repl_cost, norepl_cost)
            memoKL[(n, i)] = val
            return val

        _ = [W_KL(0, s) for s in range(K)]

        # ---------- MLE-based robust DP ----------
        def P_solver(i, Vvec, transitions_N):

            def Beta_Max(ii):
                Nrow = transitions_N[ii, :]; S = Nrow.sum()
                if ii <= len(Nrow) - 2:
                    terms = []
                    for j in range(ii, len(Nrow)):
                        terms.append(Nrow[j] * np.log(Nrow[j] / S) if Nrow[j] > 0 else 0)
                    return sum(terms)
                return 0

            BetaMax = sum(Beta_Max(r) for r in range(len(transitions_N)))

            def Lag(ii, mu, Beta_val, Vcand):
                Nrow = transitions_N[ii, :]; S = Nrow.sum()
                denom = [max(mu - Vcand[j], 1e-12) for j in range(ii, len(Nrow))]
                Lambda_inv = sum((Nrow[j] / denom[k]) for k, j in enumerate(range(ii, len(Nrow))))
                Lambda_star = (Lambda_inv)**(-1)
                inner_sum = 0.0
                for k, j in enumerate(range(ii, len(Nrow))):
                    if Nrow[j] > 0:
                        inner_sum += Nrow[j] * np.log(Nrow[j] * Lambda_star / denom[k])
                return mu - Lambda_star * (S + Beta_val) + Lambda_star * inner_sum

            df = len(transitions_N) * (len(transitions_N) - 1) / 2
            df -= count_upper_triangular_zeros(transitions_N)
            if df <= 0: df = 1
            Chq = chi2.ppf(1 - alpha, df)

            def Beta_Max_idx(ii):
                Nrow = transitions_N[ii, :]; S = Nrow.sum()
                if ii <= len(Nrow) - 2:
                    terms = []
                    for jj in range(ii, len(Nrow)):
                        terms.append(Nrow[jj] * np.log(Nrow[jj] / S) if Nrow[jj] > 0 else 0)
                    return sum(terms)
                return 0

            def muupper(ii, Vcand):
                Nrow = transitions_N[ii, :]; S = Nrow.sum()
                if S == 0: return max(Vcand) + 1.0
                Phat = Nrow / S; Vbar = np.dot(Phat, Vcand)
                Beta_i = Beta_Max_idx(ii)
                Bi_all = (BetaMax - (Chq/2)) - BetaMax + Beta_i
                num = max(Vcand) - Vbar * np.exp((Bi_all - Beta_i)/S if S>0 else 0)
                den = 1 - np.exp((Bi_all - Beta_i)/S if S>0 else 1e-12)
                den = max(den, 1e-12)
                return num / den

            def Line_Search(ii, Vcand):
                e = 1e-11
                mu_low  = max(Vcand) + e
                mu_high = muupper(ii, Vcand)
                if mu_high < mu_low: mu_high = mu_low + 1.0
                Beta_i_val = (BetaMax - (Chq/2)) - BetaMax + Beta_Max_idx(ii)
                a, b = mu_low, mu_high
                while b - a > e:
                    x1 = a + (b-a)/3; x2 = b - (b-a)/3
                    if Lag(ii, x1, Beta_i_val, Vvec) <= Lag(ii, x2, Beta_i_val, Vvec):
                        b = x2
                    else:
                        a = x1
                return Lag(ii, a, Beta_i_val, Vvec)

            if i == len(transitions_N) - 1:
                return Vvec[i]
            else:
                return Line_Search(i, Vvec)

        memoMLE = {}
        chosen_actions_matrix_MLE = np.zeros((K, T)) + 999

        def W_MLE(n, i):
            if n == T: return OC(i)
            if (n, i) in memoMLE: return memoMLE[(n, i)]
            future_vals = np.array([W_MLE(n+1, s) for s in range(K)])
            repl_cost   = R + gamma * P_solver(0, future_vals, TN)
            norepl_cost = gamma * P_solver(i, future_vals, TN)
            act = 1 if repl_cost <= norepl_cost else 0
            chosen_actions_matrix_MLE[i, n] = act
            val = OC(i) + min(repl_cost, norepl_cost)
            memoMLE[(n, i)] = val
            return val

        _ = [W_MLE(0, s) for s in range(K)]

        # ---------- Empirical MDP (from TN) ----------
        P_empirical = np.zeros((K, K), dtype=float)
        for s in range(K):
            row_sum = TN[s, :].sum()
            if row_sum == 0:
                feasible = np.arange(s, K)
                tmp = np.zeros(K); tmp[feasible] = 1.0 / len(feasible)
                P_empirical[s, :] = tmp
            else:
                P_empirical[s, :] = TN[s, :] / row_sum
        P_empirical[-1, :] = 0.0; P_empirical[-1, -1] = 1.0

        memo_emp = {}
        chosen_actions_matrix_empirical = np.zeros((K, T)) + 999

        def V_emp(n, i):
            if n == T: return OC(i)
            if (n, i) in memo_emp: return memo_emp[(n, i)]
            repl_cost   = R + gamma * sum(P_empirical[0, j] * V_emp(n+1, j) for j in range(K))
            norepl_cost = gamma * sum(P_empirical[i, j] * V_emp(n+1, j) for j in range(K))
            act = 1 if repl_cost <= norepl_cost else 0
            chosen_actions_matrix_empirical[i, n] = act
            val = OC(i) + min(repl_cost, norepl_cost)
            memo_emp[(n, i)] = val
            return val

        _ = [V_emp(0, s) for s in range(K)]

        P_empirical_final = P_empirical.copy()

        # Monte Carlo eval
        total_costs_empirical = []
        total_costs_KL        = []
        total_costs_MLE       = []

        for _path in range(n_trajectories_eval):
            c_emp = simulate_trajectory(chosen_actions_matrix_empirical, P_true)
            c_kl  = simulate_trajectory(chosen_actions_matrix_KL,        P_true)
            c_mle = simulate_trajectory(chosen_actions_matrix_MLE,       P_true)
            total_costs_empirical.append(c_emp)
            total_costs_KL.append(c_kl)
            total_costs_MLE.append(c_mle)

        alpha_values.append(alpha)
        avg_costs_empirical.append(np.mean(total_costs_empirical))
        avg_costs_KL.append(np.mean(total_costs_KL))
        avg_costs_MLE.append(np.mean(total_costs_MLE))
        Empirical_stdev_list.append(np.std(total_costs_empirical))
        KL_stdev_list.append(np.std(total_costs_KL))
        MLE_stdev_list.append(np.std(total_costs_MLE))

    # -------------------------------------------
    # Save results table
    # -------------------------------------------
    results_df = pd.DataFrame({
        'Alpha': alpha_values,
        'Empirical Avg Cost': avg_costs_empirical,
        'KL Avg Cost':        avg_costs_KL,
        'MLE Avg Cost':       avg_costs_MLE,
        'Empirical Std Dev':  Empirical_stdev_list,
        'KL Std Dev':         KL_stdev_list,
        'MLE Std Dev':        MLE_stdev_list
    })

    fname = f"Policy_performance_{component_type}_{data_case}.xlsx"
    results_df.to_excel(fname, index=False)
    print(f"Saved {fname}")

    # Debug print
    np.set_printoptions(precision=4, suppress=True)
    print("\n===== MATRICES REPORT =====")
    print(f"Component type (TRUE eval): {component_type}")
    print(f"Data case (training):       {data_case}  [source: {src_label}]")
    print("\nP_true (used to EVALUATE policies):")
    print(P_true)
    if data_case == 'unbiased':
        print("\nP_train_source (matrix that GENERATED the training data TN):")
        print(P_train_source)
    else:
        print("\nMixture weights (fragile, moderate, sturdy):", 
              None if wts is None else np.round(wts, 3))
        print("\nP_bar (simple weighted average of the three matrices):")
        print(np.round(P_bar, 4))
    print("\nP_empirical_final (estimated from TN):")
    print(np.round(P_empirical_final, 4))
    print("\nRow sums check (P_empirical_final):", 
          np.round(P_empirical_final.sum(axis=1), 6))
    print("==========================\n")

    # -------------------------------------------
    # Plot
    # -------------------------------------------
    fig_prefix = f"{component_type}_{data_case}"
    alpha_arr = np.array(alpha_values)
    avg_costs_emp_arr = np.array(avg_costs_empirical)
    avg_costs_kl_arr  = np.array(avg_costs_KL)
    avg_costs_mle_arr = np.array(avg_costs_MLE)
    std_emp_arr = np.array(Empirical_stdev_list)
    std_kl_arr  = np.array(KL_stdev_list)
    std_mle_arr = np.array(MLE_stdev_list)

    group_spacing = 2
    x_base = np.arange(len(alpha_arr)) * group_spacing
    offset = 0.35
    x_mle = x_base - offset
    x_kl  = x_base
    x_emp = x_base + offset

    fig, ax = plt.subplots(figsize=(10,6))

    ax.errorbar(x_mle, avg_costs_mle_arr, yerr=std_mle_arr, fmt='o-', capsize=4,
                elinewidth=1.5, markeredgecolor='black', label='MLE-based policy')
    ax.errorbar(x_kl,  avg_costs_kl_arr,  yerr=std_kl_arr,  fmt='s--', capsize=4,
                elinewidth=1.5, markeredgecolor='black', label='KL-based policy')
    ax.errorbar(x_emp, avg_costs_emp_arr, yerr=std_emp_arr, fmt='^-.',
                capsize=4, elinewidth=1.5, markeredgecolor='black',
                label='Empirical policy')

    subtitle = "biased training data" if data_case == 'biased' else "unbiased training data"
    ax.set_title(
        f'Policy Performance vs $\\alpha$'
        f'\n({component_type.capitalize()}, {subtitle})',
        fontsize=16
    )
    ax.set_xlabel(r'$\alpha$', fontsize=14)
    ax.set_ylabel('Total Cost (mean ± 1 SD)', fontsize=14)
    ax.grid(True, axis='y')
    ax.set_xticks(x_base)
    ax.set_xticklabels([f"{a:.2f}" for a in alpha_arr], rotation=45)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0.0, fontsize=12, frameon=False)
    plt.tight_layout()

    # ---- Filename per your request ----
    save_name = f"{fig_prefix}_NoBiase_Mixture_errorbars.png"
    plt.savefig(save_name, dpi=600, bbox_inches='tight')
    print(f"Saved figure: {save_name}")
    plt.show()

    return {
        "alpha_values":         alpha_values,
        "avg_costs_mle":        avg_costs_MLE,
        "avg_costs_kl":         avg_costs_KL,
        "avg_costs_emp":        avg_costs_empirical,
        "MLE_stdev_list":       MLE_stdev_list,
        "KL_stdev_list":        KL_stdev_list,
        "Empirical_stdev_list": Empirical_stdev_list,
        "P_true":               P_true,
        "P_empirical_final":    P_empirical_final,
        "mixture_average":      (None if data_case=='unbiased' else P_bar)
    }

########################################
# EXAMPLE USAGE
########################################

# Unbiased examples (train = eval component)
_ = run_experiment(component_type='fragile',  data_case='unbiased',
                   training_path_length=2000, seed=261, n_trajectories_eval=10000)
_ = run_experiment(component_type='moderate', data_case='unbiased',
                   training_path_length=2000, seed=262, n_trajectories_eval=10000)
_ = run_experiment(component_type='sturdy',   data_case='unbiased',
                   training_path_length=2000, seed=263, n_trajectories_eval=10000)

# Biased examples (EPISODIC mixture with equal weights by default)
_ = run_experiment(component_type='fragile',  data_case='biased',
                   training_path_length=5000, seed=264, n_trajectories_eval=10000)
_ = run_experiment(component_type='moderate', data_case='biased',
                   training_path_length=2000, seed=265, n_trajectories_eval=10000)
_ = run_experiment(component_type='sturdy',   data_case='biased',
                   training_path_length=2000, seed=266, n_trajectories_eval=10000)

# Biased example WITH NON-UNIFORM mixture weights during training
# (fragile=0.25, moderate=0.25, sturdy=0.50)
_ = run_experiment(component_type='sturdy',  data_case='biased',
                   training_path_length=2000, seed=100, n_trajectories_eval=10000,
                   mixture_weights=[0.55, 0.35, 0.10])
