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


def verify_MCTS_learning(rival_policy_class, n_games = 50, n_batches = 5):
    results_list = []
    for k in range(n_batches):
        result_dict = run_group_vs_strong(rival_policy_class, SmartMCTS, n_games=n_games, label="Group A vs SmartMCTS")
        results_list.append(result_dict)
        print(f"resultado partida {k+1}: {result_dict}")
        
    
    policy = rival_policy_class.__name__
    
    return results_list, policy, n_games
    
    
    
    

def plot_learning(match_results: list[list[dict]], n_games:int, policy_names:list[str]):
    
    plot_colors = ['r', 'g', 'b', ] #pal color de las lineas
    t_policies_col = [] #lista que almacena listas de t de cada enfrentamiento de policies
    win_policies_rates = [] # lo mismo pero con win_rates
    for policy in match_results: # match results es una lista de listas de diccionarios, es decir, una lista de juegos con una politica
        t_col = []
        win_rates = []
        for t, match in enumerate(policy): # policy es una lista de diccionarios
            agent_wins = match['strong_wins']
            total_matches = len(match['moves'])
            win_rate = agent_wins / total_matches
            print(f"Win rate for t={t+1}: {win_rate}")
            t_col.append(t+1)
            win_rates.append(win_rate)
        t_policies_col.append(t_col)
        win_policies_rates.append(win_rates)
    
    # este ciclo itera sobre las listas de listas de resultados para plotear cada una
    for t, winrate, color, policy_name in zip(t_policies_col, win_policies_rates, plot_colors, policy_names):
        plt.plot(t,winrate, color, label = policy_name)
        
        
    plt.xticks(t_policies_col[0])
    plt.ylim(0,1) # limito la escala porque los valores son de 0 a 1
    plt.title(f"Win-rate against random and Minimax policy", fontdict={"fontname": "Arial", "fontsize": 16})
    plt.xlabel(f"Batch (each = {n_games} games)")
    plt.ylabel("Win-rate")  
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    plt.show()  

def plot_agent_comparison(results_policy_A: dict, results_policy_B: dict):
    categories = ['Wins', 'Losses', 'Draws']

    agent_wins_A = results_policy_A["strong_wins"]
    agent_losses_A = results_policy_A["group_wins"]
    draws_A = results_policy_A["draws"]

    agent_wins_B = results_policy_B["strong_wins"]
    agent_losses_B = results_policy_B["group_wins"]
    draws_B = results_policy_B["draws"]

    values_A = [agent_wins_A, agent_losses_A, draws_A]
    values_B = [agent_wins_B, agent_losses_B, draws_B]


    x = np.arange(len(categories)) # [0,1,2]
    width = 0.35 # ancho de barra

    plt.figure(figsize=(10, 5))

    # Barras
    bars_A = plt.bar(x - width/2, values_A, width, label="Contra Aha")
    bars_B = plt.bar(x + width/2, values_B, width, label="Contra Minimax")

    # Títulos
    plt.title("Comparación de desempeño del agente contra politica aleatoria y Minimax", fontsize=16)
    plt.ylabel("Partidas jugadas", fontsize=14)
    plt.xticks(x+3, categories, fontsize=12)

    # Leyenda
    plt.legend(fontsize=12)

    # Líneas de valores encima de cada barra (opcional)
    for bars in [bars_A, bars_B]:
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2, 
                height + 0.1, 
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10
            )

    plt.tight_layout()
    plt.show()

        



if __name__ == "__main__":
    N_GAMES = 30  # número de partidas por grupo
    N_BATCH = 3
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
    
