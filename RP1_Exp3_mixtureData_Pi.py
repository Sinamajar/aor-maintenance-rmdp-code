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
T  = 30                              # horizon (months)
gamma = 1                            # no discounting
ALPHA_FIXED = 0.001                  # fixed alpha

#ALPHA_FIXED = 0.5 
#ALPHA_FIXED = 0.999 

R_FIXED     = 22                     # fixed replacement cost

def OC(x):
    return (36/125) * x**3           # c(0)=0, c(5)=36)

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

COMPONENTS = {'fragile': P_fragile, 'moderate': P_moderate, 'sturdy': P_sturdy}

########################################
# Utility functions
########################################

def normalize_row(p):
    p = np.asarray(p, dtype=float)
    s = p.sum()
    if s <= 0:
        return np.ones_like(p) / len(p)
    return p / s

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
    """Single-matrix generator (unbiased case).
       Replace ONLY on failure: row K-1 sends to 0 next period."""
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
    Episodic mixture generator:
      - Start at 0; pick ONE component by weights.
      - Use that SAME matrix until entering failure (state K-1).
      - On failure: reset to 0, pick a NEW component by weights.
      - Continue until 'num_steps' recorded transitions.
    """
    if seed is not None:
        np.random.seed(seed)

    if weights is None:
        weights = np.ones(len(P_list)) / len(P_list)
    weights = np.asarray(weights, dtype=float)
    weights /= weights.sum()

    Nm = P_list[0].shape[0]
    TN = np.zeros((Nm, Nm), dtype=int)

    s = 0
    comp_idx = np.random.choice(len(P_list), p=weights)
    transitions = 0

    while transitions < num_steps:
        P_use = P_list[comp_idx]
        probs = normalize_row(P_use[s])
        ns = np.random.choice(Nm, p=probs)

        TN[s, ns] += 1
        transitions += 1

        if ns == Nm - 1:
            s = 0
            comp_idx = np.random.choice(len(P_list), p=weights)
        else:
            s = ns

    # reporting-only average
    P_bar = np.tensordot(weights, np.stack(P_list, axis=0), axes=(0, 0))
    P_bar[-1, :] = 0.0; P_bar[-1, -1] = 1.0
    return TN, P_bar, weights

def q_apx_factory(TN):
    # Empirical next-state distribution per row (with your last-row reverse quirk)
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

def simulate_trajectory(chosen_actions_matrix, P_true, R_value):
    total_cost = 0.0
    s = 0
    for _t in range(T):
        a = chosen_actions_matrix[s, _t]
        total_cost += OC(s)
        if a == 1:  # replace
            total_cost += R_value
            probs = normalize_row(P_true[0])
            s = np.random.choice(K, p=probs)
        else:       # no replace
            probs = normalize_row(P_true[s])
            s = np.random.choice(K, p=probs)
    total_cost += OC(s)
    return total_cost

########################################
# Robust / Empirical DP builders
########################################

def build_KL_policy(TN, alpha, R_val):
    q_apx = q_apx_factory(TN)

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

        return line_minimizer

    memo = {}
    policy = np.zeros((K, T), dtype=int)

    def W(n, i):
        if n == T: return OC(i)
        if (n, i) in memo: return memo[(n, i)]
        future_vals = np.array([W(n+1, s) for s in range(K)])
        repl_cost   = R_val + gamma * P_solver_KL(0, future_vals)(future_vals)
        norepl_cost = gamma * P_solver_KL(i, future_vals)(future_vals)
        act = 1 if repl_cost <= norepl_cost else 0
        policy[i, n] = act
        val = OC(i) + min(repl_cost, norepl_cost)
        memo[(n, i)] = val
        return val

    _ = [W(0, s) for s in range(K)]
    return policy

def build_MLE_policy(TN, alpha, R_val):

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
            return lambda V: V[i]
        else:
            return lambda V: Line_Search(i, V)

    memo = {}
    policy = np.zeros((K, T), dtype=int)

    def W(n, i):
        if n == T: return OC(i)
        if (n, i) in memo: return memo[(n, i)]
        future_vals = np.array([W(n+1, s) for s in range(K)])
        repl_cost   = R_val + gamma * P_solver(0, future_vals, TN)(future_vals)
        norepl_cost = gamma * P_solver(i, future_vals, TN)(future_vals)
        act = 1 if repl_cost <= norepl_cost else 0
        policy[i, n] = act
        val = OC(i) + min(repl_cost, norepl_cost)
        memo[(n, i)] = val
        return val

    _ = [W(0, s) for s in range(K)]
    return policy

def build_empirical_policy(TN, R_val):
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

    memo = {}
    policy = np.zeros((K, T), dtype=int)

    def V(n, i):
        if n == T: return OC(i)
        if (n, i) in memo: return memo[(n, i)]
        repl_cost   = R_val + gamma * sum(P_empirical[0, j] * V(n+1, j) for j in range(K))
        norepl_cost = gamma * sum(P_empirical[i, j] * V(n+1, j) for j in range(K))
        act = 1 if repl_cost <= norepl_cost else 0
        policy[i, n] = act
        val = OC(i) + min(repl_cost, norepl_cost)
        memo[(n, i)] = val
        return val

    _ = [V(0, s) for s in range(K)]
    return policy, P_empirical

########################################
# NEW: sweep π_target with GROUPED error bars
########################################

def run_experiment_over_pi(component_type='fragile',
                           pis=None,
                           training_path_length=3000,
                           seed=1234,
                           n_trajectories_eval=3000):
    """
    For a given TRUE target component, fix R=22 and alpha=0.5.
    Sweep the training episodic-mixture weight π_target in [0,1].
    Remaining mass is split equally among the other two components.
    Plot grouped, side-by-side error bars per π_target.
    """
    if pis is None:
        pis = np.linspace(0.0, 1.0, 11)  # 0, 0.1, ..., 1.0

    # True environment for evaluation
    P_true = COMPONENTS[component_type].copy().astype(float)

    # Component order and list for mixture
    comp_order = ['fragile', 'moderate', 'sturdy']
    P_list = [COMPONENTS[c] for c in comp_order]
    target_idx = comp_order.index(component_type)

    # Storage
    pi_axis               = []
    avg_costs_empirical   = []
    avg_costs_KL          = []
    avg_costs_MLE         = []
    Empirical_stdev_list  = []
    KL_stdev_list         = []
    MLE_stdev_list        = []

    # Loop over π_target
    for pi in pis:
        # Mixture weights for (fragile, moderate, sturdy)
        w = np.zeros(3, dtype=float)
        w[target_idx] = pi
        rem = 1.0 - pi
        others = [i for i in range(3) if i != target_idx]
        for oi in others: w[oi] = rem / 2.0

        # ----- training TN via episodic mixture -----
        TN, _, _ = simulate_and_count_transitions_mixture_episodic(
            P_list, weights=w, num_steps=training_path_length, seed=seed
        )

        # ----- build policies (fixed alpha & R) -----
        alpha = ALPHA_FIXED
        R_val = R_FIXED

        pol_emp, _ = build_empirical_policy(TN, R_val)
        pol_kl     = build_KL_policy(TN, alpha, R_val)
        pol_mle    = build_MLE_policy(TN, alpha, R_val)

        # ----- Monte Carlo evaluation on TRUE target environment -----
        def eval_one(policy):
            total_cost = 0.0
            s = 0
            for _t in range(T):
                a = policy[s, _t]
                total_cost += OC(s)
                if a == 1:
                    total_cost += R_val
                    probs = normalize_row(P_true[0])
                    s = np.random.choice(K, p=probs)
                else:
                    probs = normalize_row(P_true[s])
                    s = np.random.choice(K, p=probs)
            total_cost += OC(s)
            return total_cost

        tot_emp, tot_kl, tot_mle = [], [], []
        for _ in range(n_trajectories_eval):
            tot_emp.append(eval_one(pol_emp))
            tot_kl.append(eval_one(pol_kl))
            tot_mle.append(eval_one(pol_mle))

        # record stats
        pi_axis.append(pi)
        avg_costs_empirical.append(np.mean(tot_emp))
        avg_costs_KL.append(np.mean(tot_kl))
        avg_costs_MLE.append(np.mean(tot_mle))
        Empirical_stdev_list.append(np.std(tot_emp))
        KL_stdev_list.append(np.std(tot_kl))
        MLE_stdev_list.append(np.std(tot_mle))

    # -------------------------------------------
    # Save results table
    # -------------------------------------------
    results_df = pd.DataFrame({
        'pi_target':           pi_axis,
        'Empirical Avg Cost':  avg_costs_empirical,
        'KL Avg Cost':         avg_costs_KL,
        'MLE Avg Cost':        avg_costs_MLE,
        'Empirical Std Dev':   Empirical_stdev_list,
        'KL Std Dev':          KL_stdev_list,
        'MLE Std Dev':         MLE_stdev_list
    })

    fname = f"Policy_performance_vsPI_{component_type}_episodic_mixture_R{R_FIXED}_alpha{ALPHA_FIXED:.2f}.xlsx"
    results_df.to_excel(fname, index=False)
    print(f"Saved {fname}")

    # -------------------------------------------
    # GROUPED error-bar plot (side-by-side)
    # -------------------------------------------
    pi_arr = np.array(pi_axis)
    avg_emp = np.array(avg_costs_empirical)
    avg_kl  = np.array(avg_costs_KL)
    avg_mle = np.array(avg_costs_MLE)
    std_emp = np.array(Empirical_stdev_list)
    std_kl  = np.array(KL_stdev_list)
    std_mle = np.array(MLE_stdev_list)

    # group layout
    group_spacing = 2.5
    x_base = np.arange(len(pi_arr)) * group_spacing
    offset = 0.35
    x_mle = x_base - offset
    x_kl  = x_base
    x_emp = x_base + offset

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.errorbar(x_mle, avg_mle, yerr=std_mle, fmt='o-', capsize=4,
                elinewidth=1.5, markeredgecolor='black', label='MLE-based policy')
    ax.errorbar(x_kl,  avg_kl,  yerr=std_kl,  fmt='s--', capsize=4,
                elinewidth=1.5, markeredgecolor='black', label='KL-based policy')
    ax.errorbar(x_emp, avg_emp, yerr=std_emp, fmt='^-.', capsize=4,
                elinewidth=1.5, markeredgecolor='black', label='Empirical policy')

    ax.set_title(
        f'Policy Performance vs $\\pi_{{{component_type}}}$ (R={R_FIXED}, $\\alpha$={ALPHA_FIXED:.3f})\n'
        f'Training data: biased; Evaluation: {component_type}',
        fontsize=15
    )
    ax.set_xlabel(f'$\\pi_{{{component_type}}}$ ', fontsize=13)
    ax.set_ylabel('Total Cost (mean ± 1 SD)', fontsize=13)
    ax.grid(True, axis='y')

    ax.set_xticks(x_base)
    ax.set_xticklabels([f"{pi:.1f}" for pi in pi_arr], rotation=0)

    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0.0, fontsize=12, frameon=False)
    plt.tight_layout()

    save_name = f"{component_type}_episodic_mixture_vsPI_R{R_FIXED}_alpha{ALPHA_FIXED:.2f}_errorbars_grouped.png"
    plt.savefig(save_name, dpi=600, bbox_inches='tight')
    print(f"Saved figure: {save_name}")
    plt.show()

    return {
        "pi_values":            pi_axis,
        "avg_costs_mle":        avg_costs_MLE,
        "avg_costs_kl":         avg_costs_KL,
        "avg_costs_emp":        avg_costs_empirical,
        "MLE_stdev_list":       MLE_stdev_list,
        "KL_stdev_list":        KL_stdev_list,
        "Empirical_stdev_list": Empirical_stdev_list
    }

########################################
# EXAMPLE USAGE
########################################

if __name__ == "__main__":
    pi_grid = np.linspace(0.01, 0.99, 10)     # 0, 0.1, ..., 1.0
    training_n = 2000
    eval_paths = 10000

    _ = run_experiment_over_pi(component_type='fragile',
                               pis=pi_grid,
                               training_path_length=training_n,
                               seed=264,
                               n_trajectories_eval=eval_paths)

    _ = run_experiment_over_pi(component_type='moderate',
                               pis=pi_grid,
                               training_path_length=training_n,
                               seed=265,
                               n_trajectories_eval=eval_paths)

    _ = run_experiment_over_pi(component_type='sturdy',
                               pis=pi_grid,
                               training_path_length=training_n,
                               seed=266,
                               n_trajectories_eval=eval_paths)
