import numpy as np
import pandas as pd
import csv
from typing import Dict, Optional

# ============================================================
# Osborne (2003) Table 2 probabilities (three attacker vs two defender dice cases)
# (i attacker dice, j defender dice) -> outcomes as (Δa, Δd, p)
# Δa = attacker army losses; Δd = defender army losses
# ============================================================
T2 = {
    (1, 1): [("D-1", (0, 1), 0.417), ("A-1", (1, 0), 0.583)],
    (1, 2): [("D-1", (0, 1), 0.255), ("A-1", (1, 0), 0.745)],
    (2, 1): [("D-1", (0, 1), 0.579), ("A-1", (1, 0), 0.421)],
    (2, 2): [("D-2", (0, 2), 0.228), ("each-1", (1, 1), 0.324), ("A-2", (2, 0), 0.448)],
    (3, 1): [("D-1", (0, 1), 0.660), ("A-1", (1, 0), 0.340)],
    (3, 2): [("D-2", (0, 2), 0.372), ("each-1", (1, 1), 0.336), ("A-2", (2, 0), 0.293)],
}

def _outcome_probs(i, j):
    """Return list of (Δa, Δd, p) given i attacker dice and j defender dice."""
    return [(da, dd, p) for _, (da, dd), p in T2[(i, j)]]

# ============================================================
# Dice policies (defaults reproduce Osborne's "full force")
# Provide custom callables (a,d) -> dice count if you want other strategies.
# ============================================================
def attacker_policy_full(a, d):
    """Default: roll as many dice as allowed by a (max 3, min 1)."""
    return max(1, min(3, a))

def defender_policy_full(a, d):
    """Default: roll as many dice as allowed by d (max 2, min 1)."""
    return max(1, min(2, d))

# Example optional policies you can pass in:
def attacker_conservative(a, d):
    """Example: reduce dice when low / facing 2 dice defender."""
    if a >= 3 and d >= 2:
        return 2
    if a == 2:
        return 1
    return 1

def defender_roll_one_if_two_left(a, d):
    """Example: defender rolls 1 die when d==2 to avoid 2-loss outcomes."""
    return 1 if d == 2 else max(1, min(2, d))

# ============================================================
# State ordering (Osborne)
# Transient: (1,1)...(1,D),(2,1)...(A,D)  [A*D states]
# Absorbing: (0,1)...(0,D), (1,0)...(A,0) [A+D states]
# ============================================================
def _state_order(A, D):
    transient = [(a, d) for a in range(1, A + 1) for d in range(1, D + 1)]
    absorbing = [(0, d) for d in range(1, D + 1)] + [(a, 0) for a in range(1, A + 1)]
    return transient + absorbing

# ============================================================
# Build full transition matrix P (policy-enabled)
# ============================================================
def _build_P(A, D, attacker_policy, defender_policy):
    states = _state_order(A, D)
    idx = {s: i for i, s in enumerate(states)}
    n = len(states)
    P = np.zeros((n, n))

    # Transient rows
    for a in range(1, A + 1):
        for d in range(1, D + 1):
            row = idx[(a, d)]
            i = max(1, min(3, attacker_policy(a, d), a))
            j = max(1, min(2, defender_policy(a, d), d))
            if (i, j) not in T2:
                raise ValueError(f"No Table 2 probabilities for dice combo (i={i}, j={j}).")
            for da, dd, p in _outcome_probs(i, j):
                na, nd = max(0, a - da), max(0, d - dd)
                P[row, idx[(na, nd)]] += p

    # Absorbing rows = identity
    for s in states[A * D:]:
        P[idx[s], idx[s]] = 1.0

    return states, P

def _split_QR(P, A, D):
    nT = A * D
    Q = P[:nT, :nT]
    R = P[:nT, nT:]
    return Q, R

def _F_from_QR(Q, R):
    I = np.eye(Q.shape[0])
    N = np.linalg.inv(I - Q)
    return N @ R  # F = (I-Q)^(-1) R

# ============================================================
# Public APIs
# ============================================================


def player1_win_prob(
    A, D,
    attacker_policy=attacker_policy_full,
    defender_policy=defender_policy_full,
):
    """Return Pr(attacker wins) for given (A, D)."""
    states, P = _build_P(A, D, attacker_policy, defender_policy)
    Q, R = _split_QR(P, A, D)
    F = _F_from_QR(Q, R)
    row = F[A * D - 1]
    return float(row[D:].sum())  # last A absorbing columns: (1,0)...(A,0)

def expected_losses(
    A, D,
    attacker_policy=attacker_policy_full,
    defender_policy=defender_policy_full,
):
    """Return expected losses: {'attacker': E[LA], 'defender': E[LD]}."""
    states, P = _build_P(A, D, attacker_policy, defender_policy)
    Q, R = _split_QR(P, A, D)
    F = _F_from_QR(Q, R)
    row = F[A * D - 1]

    pr_rd = row[:D]     # defender remaining distribution
    pr_ra = row[D:]     # attacker remaining distribution
    rd_vals = np.arange(1, D + 1)
    ra_vals = np.arange(1, A + 1)

    E_RD = float(np.dot(rd_vals, pr_rd))
    E_RA = float(np.dot(ra_vals, pr_ra))

    return {"attacker": A - E_RA, "defender": D - E_RD}

def odds_grid(
    max_A, max_D,
    attacker_policy=attacker_policy_full,
    defender_policy=defender_policy_full,
):
    """Return DataFrame of Pr(attacker wins) for A=1..max_A, D=1..max_D."""
    data = []
    for A in range(1, max_A + 1):
        row = []
        for D in range(1, max_D + 1):
            row.append(player1_win_prob(A, D, attacker_policy, defender_policy))
        data.append(row)
    return pd.DataFrame(
        data,
        index=[f"A={i}" for i in range(1, max_A + 1)],
        columns=[f"D={j}" for j in range(1, max_D + 1)],
    )

def _absorption_row(A, D, attacker_policy, defender_policy):
    """Internal helper: return the absorption probability row for start (A,D)."""
    states, P = _build_P(A, D, attacker_policy, defender_policy)
    Q, R = _split_QR(P, A, D)
    F = _F_from_QR(Q, R)
    return F[A * D - 1], F  # row for (A,D), and full F if caller wants it

def expected_losses_with_std(
    A, D,
    attacker_policy=attacker_policy_full,
    defender_policy=defender_policy_full,
):
    """
    Return mean and standard deviation of losses for attacker and defender:
      {
        'attacker': {'mean': E[LA], 'std': SD[LA]},
        'defender': {'mean': E[LD], 'std': SD[LD]}
      }
    Uses F = (I-Q)^(-1) R to get absorption distributions, then computes
    E[R•], Var[R•], and converts to losses.
    """
    row, _ = _absorption_row(A, D, attacker_policy, defender_policy)

    # Absorbing columns: first D are (0,1)...(0,D) => defender remaining = 1..D
    #                    last A are (1,0)...(A,0) => attacker remaining = 1..A
    pr_rd = row[:D]
    pr_ra = row[D:]

    rd_vals = np.arange(1, D + 1, dtype=float)  # possible R_D
    ra_vals = np.arange(1, A + 1, dtype=float)  # possible R_A

    # Moments for R_D
    E_RD  = float(np.dot(rd_vals, pr_rd))
    E_RD2 = float(np.dot(rd_vals**2, pr_rd))
    Var_RD = max(0.0, E_RD2 - E_RD**2)
    SD_RD  = Var_RD**0.5

    # Moments for R_A
    E_RA  = float(np.dot(ra_vals, pr_ra))
    E_RA2 = float(np.dot(ra_vals**2, pr_ra))
    Var_RA = max(0.0, E_RA2 - E_RA**2)
    SD_RA  = Var_RA**0.5

    # Convert to losses: L_D = D - R_D, L_A = A - R_A
    E_LD = D - E_RD
    E_LA = A - E_RA
    SD_LD = SD_RD  # Var(L) = Var(R)
    SD_LA = SD_RA

    return {
        "attacker": {"mean": E_LA, "std": SD_LA},
        "defender": {"mean": E_LD, "std": SD_LD},
    }

# --- Optional: augment battle_summary to include standard deviations too ---
def battle_summary(
    A, D,
    as_dataframes=True,
    attacker_policy=attacker_policy_full,
    defender_policy=defender_policy_full,
):
    states, P = _build_P(A, D, attacker_policy, defender_policy)
    Q, R = _split_QR(P, A, D)
    F = _F_from_QR(Q, R)

    # Starting from (A,D): it's the last transient row -> index A*D - 1
    last_row = F[A * D - 1]
    defender_abs_cols = slice(0, D)           # (0,1)...(0,D)
    attacker_abs_cols = slice(D, D + A)       # (1,0)...(A,0)

    p_defender_wins = float(last_row[defender_abs_cols].sum())
    p_attacker_wins = float(last_row[attacker_abs_cols].sum())

    rd_k = np.arange(1, D + 1, dtype=float)
    ra_k = np.arange(1, A + 1, dtype=float)
    pr_rd = last_row[defender_abs_cols]
    pr_ra = last_row[attacker_abs_cols]

    # Expected remaining
    E_RD = float(np.dot(rd_k, pr_rd))
    E_RA = float(np.dot(ra_k, pr_ra))
    # Second moments
    E_RD2 = float(np.dot(rd_k**2, pr_rd))
    E_RA2 = float(np.dot(ra_k**2, pr_ra))
    # Variances
    Var_RD = max(0.0, E_RD2 - E_RD**2)
    Var_RA = max(0.0, E_RA2 - E_RA**2)

    # Convert to losses
    E_LD = D - E_RD
    E_LA = A - E_RA
    SD_LD = Var_RD**0.5
    SD_LA = Var_RA**0.5

    results = {
        "states": states,
        "P": P, "Q": Q, "R": R, "F": F,
        "p_attacker_wins": p_attacker_wins,
        "p_defender_wins": p_defender_wins,
        "expected_losses": {"attacker": float(E_LA), "defender": float(E_LD)},
        "std_losses": {"attacker": float(SD_LA), "defender": float(SD_LD)},
        "RA_distribution": dict(zip(map(int, ra_k), map(float, pr_ra))),
        "RD_distribution": dict(zip(map(int, rd_k), map(float, pr_rd))),
    }

    if as_dataframes:
        state_labels = [f"({a},{d})" for (a, d) in states]
        nT = A * D
        R_cols = [f"(0,{d})" for d in range(1, D + 1)] + [f"({a},0)" for a in range(1, A + 1)]
        results["P_df"] = pd.DataFrame(P, index=state_labels, columns=state_labels)
        results["Q_df"] = pd.DataFrame(Q, index=state_labels[:nT], columns=state_labels[:nT])
        results["R_df"] = pd.DataFrame(R, index=state_labels[:nT], columns=R_cols)
        results["F_df"] = pd.DataFrame(F, index=state_labels[:nT], columns=R_cols)

    return results


# Extension
# --- Plateau detection helpers -----------------------------------------


def compute_conquest_prob_grid(
    A_max: int,
    D_max: int,
    attacker_policy: str = "full_force",
    defender_policy: str = "full_force",
) -> np.ndarray:
    """
    Build a grid P[a,d] = P(attacker_conquers | a attacker troops, d defender troops)
    using the existing Markov machinery (battle_summary).

    a, d indices are 0-based: P[a,d] corresponds to (A=a, D=d).
    P[0, :] and P[:, 0] follow the boundary conditions implied by battle_summary.
    """
    P = np.zeros((A_max + 1, D_max + 1), dtype=float)

    # Boundary: defender already dead -> attacker has already "won".
    P[:, 0] = 1.0

    for a in range(1, A_max + 1):
        for d in range(1, D_max + 1):
            summary = battle_summary(
                A=a,
                D=d,
                attacker_policy=attacker_policy,
                defender_policy=defender_policy,
            )
            # summary["p_attacker_wins"] should already be in your API;
            # if the name differs, adjust this line.
            p_win = summary["p_attacker_wins"]
            P[a, d] = float(p_win)

    return P


def compute_plateau_thresholds(
    A_max: int,
    D_max: int,
    eps: float = 1e-3,
    min_streak: int = 3,
    attacker_policy: str = "full_force",
    defender_policy: str = "full_force",
) -> Dict[int, Optional[int]]:
    """
    For each defender troop count d in [1..D_max], find the smallest A such that
    incremental gains in P_win(A,d) become negligible and stay that way:

        |P(A+1,d) - P(A,d)| < eps   for `min_streak` consecutive steps.

    Returns a dict: plateau[d] = A_plateau or None if not found before A_max.

    This is meant to guide:
      - where we can treat extra attacker troops as 'reserves' that don't change
        the action profile much,
      - where we are in a sensitive regime that must be solved exactly.
    """
    P = compute_conquest_prob_grid(
        A_max=A_max,
        D_max=D_max,
        attacker_policy=attacker_policy,
        defender_policy=defender_policy,
    )

    plateau: Dict[int, Optional[int]] = {}

    for d in range(1, D_max + 1):
        A_plateau: Optional[int] = None
        streak = 0
        for a in range(1, A_max):
            delta = abs(P[a + 1, d] - P[a, d])
            if delta < eps:
                streak += 1
                if streak >= min_streak:
                    # plateau starts at the first a in the streak
                    A_plateau = a - (min_streak - 1)
                    break
            else:
                streak = 0
        plateau[d] = A_plateau

    return plateau




# ============================================================
# Summary
# ============================================================
if __name__ == "__main__":
    #Single (A,D)
    A, D = 2, 2
    res = battle_summary(A, D)
    # print(f"A={A}, D={D}")
    # print(f"Attacker win probability: {res['p_attacker_wins']:.9f}")
    # print("Expected losses:", res["expected_losses"])
    # print("Std of losses:", res["std_losses"])
    # print("\nF = (I-Q)^(-1) R (absorption probs from each transient state):")
    # print(res["F_df"])
    # print(res["F_df"].iloc[5,:])
    # res["F_df"].to_csv("combat_df.csv", index=True)
    print(res)









