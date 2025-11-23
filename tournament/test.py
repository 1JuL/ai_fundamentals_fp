# test.py
import numpy as np
import multiprocessing
import random
import os
from tqdm import tqdm

from connect4.connect_state import ConnectState
from connect4.policy import Policy

# Policy fuerte (MCTS) definida en policy.py en la raíz
from policy import SmartMCTS

# Policies de los grupos (todas random)
from groups.GroupA.policy import Aha
from groups.GroupB.policy import Hello
from groups.GroupC.policy import OhYes
from groups.GroupD.policy import ClaimEvenPolicy
from groups.GroupF.policy import MinimaxPolicy
from groups.GroupE.policy import AllisPolicy

def play_single_game(policy_neg: Policy, policy_pos: Policy, render: bool = False):
    state = ConnectState()
    n_moves = 0

    policies = {-1: policy_neg, 1: policy_pos}

    while not state.is_final():
        current_player = state.player
        current_policy = policies[current_player]

        action = current_policy.act(state.board)

        if not state.is_applicable(action):
            winner = -current_player
            return winner, n_moves

        state = state.transition(action)
        n_moves += 1

        if render:
            print(state.board)
            print("------------------------")

    winner = state.get_winner()
    return winner, n_moves


def run_game(group_policy_class, strong_policy_class, starts_first, seed):
    # Semilla para los random internos de esta partida
    random.seed(seed)
    np.random.seed(int.from_bytes(seed, "little", signed=False))

    group = group_policy_class()
    group.mount()

    strong = strong_policy_class()
    strong.mount()

    if starts_first:
        return play_single_game(group, strong)
    else:
        return play_single_game(strong, group)


# Helper para poder usar imap_unordered (acepta un solo argumento)
def run_game_star(args):
    return run_game(*args)


def run_group_vs_strong(group_policy_class, strong_policy_class, n_games: int = 1, label: str = "Simulating games"):
    results_dict = {
        "group_wins": 0,
        "strong_wins": 0,
        "draws": 0,
        "moves": [],
    }

    # Preparamos argumentos: alternamos quién empieza y generamos semillas únicas
    args = [
        (group_policy_class, strong_policy_class, i % 2 == 0, os.urandom(4))
        for i in range(n_games)
    ]

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        # AHORA SÍ: tqdm envuelve a un iterador (imap_unordered), no a starmap
        results = []
        for res in tqdm(
            pool.imap_unordered(run_game_star, args),
            total=n_games,
            desc=label
        ):
            results.append(res)

    for i, (winner, n_moves) in enumerate(results):
        # i par: grupo empezó (-1); i impar: empezó SmartMCTS (-1)
        if i % 2 == 0:
            # Grupo fue jugador -1
            if winner == -1:
                results_dict["group_wins"] += 1
            elif winner == 1:
                results_dict["strong_wins"] += 1
            else:
                results_dict["draws"] += 1
        else:
            # SmartMCTS fue jugador -1
            if winner == -1:
                results_dict["strong_wins"] += 1
            elif winner == 1:
                results_dict["group_wins"] += 1
            else:
                results_dict["draws"] += 1

        results_dict["moves"].append(n_moves)

    results_dict["mean_moves"] = float(np.mean(results_dict["moves"])) if results_dict["moves"] else 0.0
    return results_dict


if __name__ == "__main__":
    N = 10  # número de partidas por grupo

    print("\nGroup A (Aha) vs SmartMCTS ------------------------")
    print(run_group_vs_strong(Aha, SmartMCTS, n_games=N, label="Group A vs SmartMCTS"))

    print("\nGroup B (Hello) vs SmartMCTS ----------------------")
    print(run_group_vs_strong(Hello, SmartMCTS, n_games=N, label="Group B vs SmartMCTS"))

    print("\nGroup C (OhYes) vs SmartMCTS ----------------------")
    print(run_group_vs_strong(OhYes, SmartMCTS, n_games=N, label="Group C vs SmartMCTS"))

    print("\nGroup D (ClaimEvenPolicy) vs SmartMCTS ----------------------")
    print(run_group_vs_strong(ClaimEvenPolicy, SmartMCTS, n_games=N, label="Group D vs SmartMCTS")) 

    print("\nGroup F (MinimaxPolicy) vs SmartMCTS ----------------------")
    print(run_group_vs_strong(MinimaxPolicy, SmartMCTS, n_games=N, label="Group F vs SmartMCTS"))

    print("\nGroup E (Alli Policy) vs SmartMCTS ----------------------")
    print(run_group_vs_strong(AllisPolicy, SmartMCTS, n_games=N, label="Group E vs SmartMCTS"))
