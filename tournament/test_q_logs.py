# test.py
import numpy as np
import multiprocessing
import random
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import inspect
import pandas as pd

from connect4.connect_state import ConnectState
from connect4.policy import Policy
from policy import SmartMCTS

# Groups (random/baseline opponents)
from groups.GroupA.policy import Aha
from groups.GroupB.policy import Hello
from groups.GroupC.policy import OhYes
from groups.GroupD.policy import ClaimEvenPolicy
from groups.GroupF.policy import MinimaxPolicy
from groups.GroupE.policy import AllisPolicy


# add near your imports in test.py
import glob

def merge_q_parquets(target_qpath: str, remove_pid_files: bool = True) -> None:
    """
    Merge existing target_qpath (if present) and all per-pid files:
      {target_qpath}.pid*.parquet
    For each (state,action) we compute:
      combined_count = sum(counts)
      combined_q = sum(q_i * count_i) / combined_count
    Writes atomically to target_qpath.
    """
    all_dfs = []

    # load existing global q if present
    if os.path.exists(target_qpath):
        try:
            df_base = pd.read_parquet(target_qpath)
            if not df_base.empty:
                all_dfs.append(df_base)
        except Exception as e:
            print("Warning: failed reading existing global q file:", e)

    pid_pattern = f"{target_qpath}.pid*.parquet"
    pid_files = glob.glob(pid_pattern)
    for pf in pid_files:
        try:
            df_pf = pd.read_parquet(pf)
            if not df_pf.empty:
                all_dfs.append(df_pf)
        except Exception as e:
            print(f"Warning: failed reading pid file {pf}: {e}")

    if not all_dfs:
        # nothing to merge; ensure there is an (empty) global file if missing
        if not os.path.exists(target_qpath):
            df_empty = pd.DataFrame([], columns=["state", "action", "q", "sa_count"])
            tmp = f"{target_qpath}.tmp"
            df_empty.to_parquet(tmp, index=False)
            os.replace(tmp, target_qpath)
        return

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all["action"] = df_all["action"].astype(int)
    df_all["q"] = df_all["q"].astype(float)
    df_all["sa_count"] = df_all["sa_count"].astype(int)

    # group and compute weighted average
    grouped = (
        df_all.groupby(["state", "action"], as_index=False)
        .apply(lambda g: pd.Series({
            "q": (g["q"] * g["sa_count"]).sum() / g["sa_count"].sum(),
            "sa_count": int(g["sa_count"].sum())
        }))
        .reset_index()
    )

    grouped = grouped[["state", "action", "q", "sa_count"]]

    tmp = f"{target_qpath}.tmp"
    grouped.to_parquet(tmp, index=False)
    os.replace(tmp, target_qpath)
    print(f"Merged {len(pid_files)} pid-files into {target_qpath} (rows={len(grouped)})")

    if remove_pid_files:
        for pf in pid_files:
            try:
                os.remove(pf)
            except Exception:
                pass




# ============================================================
# Helper: supports mount() with or without timeout
# ============================================================
def safe_mount(policy: Policy, time_out: int = 10):
    try:
        sig = inspect.signature(policy.mount)
        if len(sig.parameters) >= 2:
            policy.mount(time_out)
        else:
            policy.mount()
    except Exception:
        try:
            policy.mount()
        except Exception:
            pass


# ============================================================
# IMPORTANT: Create SmartMCTS ONCE per worker process
# ============================================================
_worker_strong_policy = None  # persistent MCTS inside the worker


def init_worker(strong_policy_class, q_path):
    global _worker_strong_policy
    _worker_strong_policy = strong_policy_class(q_path=q_path)
    _worker_strong_policy._autosave_parquet = True
    safe_mount(_worker_strong_policy, 10)


# ============================================================
# Play ONE game between:
#      - group policy (fresh policy each game)
#      - persistent SmartMCTS (per worker)
# ============================================================
def play_single_game(policy_neg: Policy, policy_pos: Policy, render: bool = False):
    state = ConnectState()
    n_moves = 0

    policies = {-1: policy_neg, 1: policy_pos}

    while not state.is_final():
        current_player = state.player
        current_policy = policies[current_player]

        action = current_policy.act(state.board)

        if not state.is_applicable(action):
            return -current_player, n_moves  # illegal move => opponent wins

        state = state.transition(action)
        n_moves += 1

        if render:
            print(state.board)
            print("------------------------")

    return state.get_winner(), n_moves


def run_game(group_policy_class, starts_first, seed):
    """
    group: recreated each game (they don’t learn)
    smartMCTS: persistent (keeps its Q-table)
    """
    global _worker_strong_policy

    random.seed(seed)
    np.random.seed(int.from_bytes(seed, "little", signed=False))

    group = group_policy_class()
    safe_mount(group, 10)

    strong = _worker_strong_policy  # persistent MCTS inside process

    # Decide order
    if starts_first:
        return play_single_game(group, strong)
    else:
        return play_single_game(strong, group)


def run_game_star(args):
    return run_game(*args)


# ============================================================
# Simulate full batch
# ============================================================
def run_group_vs_strong(group_policy_class, strong_policy_class, n_games: int, q_path, label: str):
    results = {
        "group_wins": 0,
        "strong_wins": 0,
        "draws": 0,
        "moves": []
    }

    args = [
        (group_policy_class, i % 2 == 0, os.urandom(4))
        for i in range(n_games)
    ]

    with multiprocessing.Pool(
        processes=multiprocessing.cpu_count(),
        initializer=init_worker,
        initargs=(strong_policy_class, q_path,)
    ) as pool:

        for i, (winner, n_moves) in enumerate(
                tqdm(pool.imap_unordered(run_game_star, args), total=n_games, desc=label)
        ):
            if i % 2 == 0:  # group was -1
                if winner == -1:
                    results["group_wins"] += 1
                elif winner == 1:
                    results["strong_wins"] += 1
                else:
                    results["draws"] += 1
            else:           # strong was -1
                if winner == -1:
                    results["strong_wins"] += 1
                elif winner == 1:
                    results["group_wins"] += 1
                else:
                    results["draws"] += 1

            results["moves"].append(n_moves)

    results["mean_moves"] = float(np.mean(results["moves"])) if results["moves"] else 0
    return results


# ============================================================
# Learning test
# ============================================================
def verify_MCTS_learning(n_games=50, n_batchs=5, rival_policy_class=Aha, q_path="q_values.parquet"):
    results_list = []

    for k in range(n_batchs):
        result = run_group_vs_strong(
            rival_policy_class,
            SmartMCTS,
            n_games=n_games,
            q_path=q_path,
            label=f"Batch {k+1}/{n_batchs}"
        )

        # MERGE worker pid files into the global q_values.parquet
        try:
            merge_q_parquets(q_path, remove_pid_files=True)
        except Exception as e:
            print("Warning: merge_q_parquets() failed:", e)

        results_list.append(result)
        print(f"Batch {k+1} result: {result}")

    plot_learning(results_list, rival_policy_class.__name__, n_games)



def plot_learning(match_results, policy_name, n_games):
    t_col = []
    win_rates = []

    for t, match in enumerate(match_results):
        strong_wins = match["strong_wins"]
        total = strong_wins + match["group_wins"] + match["draws"]
        win_rate = strong_wins / total
        t_col.append(t + 1)
        win_rates.append(win_rate)

    plt.plot(t_col, win_rates, "r")
    plt.xticks(t_col)
    plt.ylim(0, 1)
    plt.title(f"Win-rate vs {policy_name}")
    plt.xlabel(f"Batch (each = {n_games} games)")
    plt.ylabel("Win-rate")
    plt.show()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    N_GAMES = 3
    N_BATCH = 1
    RIVAL = SmartMCTS
    Q_PATH = "q_values.parquet"

    print(f"=== Running {N_BATCH} batches of {N_GAMES} games vs {RIVAL.__name__} ===")
    verify_MCTS_learning(N_GAMES, N_BATCH, RIVAL, q_path=Q_PATH)
