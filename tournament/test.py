# test.py
import numpy as np
import multiprocessing
import random
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import inspect  # NUEVO: para detectar si mount recibe timeout

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


# NUEVO:
# Helper para soportar ambos tipos de mount:
# - mount(self)
# - mount(self, time_out)
def safe_mount(policy: Policy, time_out: int = 10):
    try:
        sig = inspect.signature(policy.mount)
        # si tiene 2 parámetros (self, time_out) => le pasamos timeout
        if len(sig.parameters) >= 2:
            policy.mount(time_out)
        else:
            policy.mount()
    except Exception:
        # si algo raro pasa, intentamos mount sin args para no romper test local
        try:
            policy.mount()
        except Exception:
            pass


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
    safe_mount(group, 10)  # NUEVO: soporta mount con/sin timeout

    strong = strong_policy_class()
    safe_mount(strong, 10)  # NUEVO: SmartMCTS sí usa timeout

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


def verify_MCTS_learning(n_games = 50, n_batchs = 5, rival_policy_class = Aha):
    results_list = []
    for k in range(n_batchs):
        result_dict = run_group_vs_strong(rival_policy_class, SmartMCTS, n_games=n_games, label="Group A vs SmartMCTS")
        results_list.append(result_dict)
        print(f"resultado partida {k+1}: {result_dict}")
        
    
    policy = rival_policy_class.__name__
    plot_learning(results_list, policy, n_games)
    
    

def plot_learning(match_results: list[dict], policy_name:str, n_games:int):
    t_col = []
    win_rates = []
    
    for t, match in enumerate(match_results):
        agent_wins = match['strong_wins']
        total_matches = len(match['moves'])
        win_rate = agent_wins / total_matches
        t_col.append(t+1)
        win_rates.append(win_rate)
    
    plt.plot(t_col,win_rates, 'r')
    plt.xticks(t_col)
    plt.ylim(0,1) # limito la escala porque los valores son de 0 a 1
    plt.title(f"Tasa de partidas ganadas contra {policy_name}", fontdict={"fontname": "Arial", "fontsize": 16})
    plt.xlabel(f"Catidad de lotes de {n_games} partidas jugados")
    plt.ylabel("Tasa de partidas ganadas")  
    
    plt.show()  
        



if __name__ == "__main__":
    N_GAMES = 150  # número de partidas por grupo
    N_BATCH = 5
    RIVAL_POLICY = Aha

    # print("\nGroup A (Aha) vs SmartMCTS ------------------------")
    # print(run_group_vs_strong(Aha, SmartMCTS, n_games=N, label="Group A vs SmartMCTS"))

    # print("\nGroup B (Hello) vs SmartMCTS ----------------------")
    # print(run_group_vs_strong(Hello, SmartMCTS, n_games=N, label="Group B vs SmartMCTS"))

    # print("\nGroup C (OhYes) vs SmartMCTS ----------------------")
    # print(run_group_vs_strong(OhYes, SmartMCTS, n_games=N, label="Group C vs SmartMCTS"))

    # print("\nGroup D (ClaimEvenPolicy) vs SmartMCTS ----------------------")
    # print(run_group_vs_strong(ClaimEvenPolicy, SmartMCTS, n_games=N, label="Group D vs SmartMCTS")) 

    # print("\nGroup F (MinimaxPolicy) vs SmartMCTS ----------------------")
    # print(run_group_vs_strong(MinimaxPolicy, SmartMCTS, n_games=N, label="Group F vs SmartMCTS"))

    # print("\nGroup E (Alli Policy) vs SmartMCTS ----------------------")
    # print(run_group_vs_strong(AllisPolicy, SmartMCTS, n_games=N, label="Group E vs SmartMCTS"))
    
    print(F"=== Ejecutando prueba de {N_BATCH} lotes de {N_GAMES} patidas contra {RIVAL_POLICY.__name__} ===")
    verify_MCTS_learning(N_GAMES,N_BATCH, rival_policy_class=Aha)
    
