import numpy as np
import math
import random
import json
import os
import time  # NUEVO: para medir deadline global dentro de act y evitar timeouts
from connect4.policy import Policy

from mctsnode import MCTSNode

from helpers import (
    ROWS, COLS,
    ordered_moves, get_heights,
    creates_odd_threat, claims_even, is_followup_threat,
    opponent_can_respond, is_useful_threat,
    heuristic, find_winning_move, is_block_move, creates_useful_threat
)

from board_utils import (
    get_winner, is_terminal, 
    get_legal_moves, simulate_move, get_current_player
)

from tactical_search import (
    quick_minimax, min_value, max_value
)

move_order = [3, 2, 4, 1, 5, 0, 6]  # centro primero
win_value = 1_000_000
window_scores = {4: win_value, 3: 5000, 2: 200, 1: 10}

def generate_trial(
    state: np.ndarray,
    root_player: int,
    max_trial_depth: int,
    internal_iterations: int,
    deadline: float,
    c_param: float,
    max_rollout_depth: int
) -> tuple[list[tuple[np.ndarray, int, int]], np.ndarray]:
    trial: list[tuple[np.ndarray, int, int]] = []
    current_state = state.copy()
    current_player = root_player

    for _ in range(max_trial_depth):
        if is_terminal(current_state):
            break

        legal_moves = get_legal_moves(current_state)
        if not legal_moves:
            break

        # PODA EN TRIAL - Repite la lógica useful_cols dentro del trial.

        heights = get_heights(current_state)
        parities = [h % 2 for h in heights]
        useful_cols = [
            c for c in legal_moves
            if is_useful_threat(current_state, c, current_player, heights, parities)
        ]
        pruned_moves = useful_cols if useful_cols else legal_moves
        pruned_moves = ordered_moves(pruned_moves, move_order)

        q_prime = mini_mcts(current_state, current_player, internal_iterations, deadline, c_param, max_rollout_depth)
        action = select_from_qprime(q_prime, pruned_moves)

        trial.append((current_state.copy(), action, current_player))

        current_state = simulate_move(current_state, action, current_player)
        current_player = -current_player

    return trial, current_state

def evaluate_trial(final_state: np.ndarray, root_player: int) -> float:
    winner = get_winner(final_state)
    if winner == root_player:
        return 1.0
    elif winner == -root_player:
        return -1.0

    # Heurística de Minimax
    # Si el trial termina sin ganador, no recompensamos con 0, usamos una heurística tipo minimax (ventanas propias - rival).

    h = heuristic(final_state, root_player, window_scores)
    return float(np.tanh(h / 5000.0))

def update_q_global(
    trial: list[tuple[np.ndarray, int, int]],
    reward: float,
    root_player: int,
    block_bonus: float,
    threat_bonus: float,
    center_bonus: float,
    sa_counts: dict[str, int],
    q_table: dict[str, float],
) -> None:

    for state, action, player_at_state in trial:
        target = reward if player_at_state == root_player else -reward

        # Rewards adicionales
        #  block_bonus si la acción bloquea victoria rival inmediata
        #  threat_bonus si crea amenaza útil
        #  center_bonus por jugar en columna 3

        shap = 0.0

        if is_block_move(
            state, action, player_at_state,
            get_legal_moves, simulate_move, get_winner
        ):
            shap += block_bonus

        if creates_useful_threat(
            state, action, player_at_state,
            simulate_move, window_scores
        ):
            shap += threat_bonus

        if action == 3:
            shap += center_bonus

        target = target + shap

        key = sa_key(state, action)
        n = sa_counts.get(key, 0) + 1
        sa_counts[key] = n

        old_q = q_table.get(key, 0.0)
        new_q = old_q + (target - old_q) / n
        q_table[key] = new_q

def mini_mcts(state: np.ndarray, current_player: int, internal_iterations: int, deadline: float, c_param: float, max_rollout_depth: int) -> dict[int, float]:
    root = MCTSNode(state.copy(), -current_player)

    for _ in range(internal_iterations):
        # NUEVO: corte por tiempo dentro de MCTS interno
        if deadline is not None and time.perf_counter() > deadline:
            break

        node = root
        sim_state = state.copy()

        while node.children and not is_terminal(sim_state):
            node = node.best_child(c_param)
            sim_state = simulate_move(sim_state, node.action, -node.parent.player)

        if not is_terminal(sim_state):
            current_legal = get_legal_moves(sim_state)
            if current_legal:

                # Centro de expansión: Ordena jugadas al centro y elige aleatorio entre top3.

                current_legal = ordered_moves(current_legal, move_order)
                action = random.choice(current_legal[:3])

                new_state = simulate_move(sim_state, action, -node.player)
                node = node.expand(action, new_state, -node.player)
                sim_state = new_state

        sim_player = -node.player
        depth = 0
        while not is_terminal(sim_state) and depth < max_rollout_depth:
            # NUEVO: corte por tiempo dentro de rollout
            if deadline is not None and time.perf_counter() > deadline:
                break

            sim_actions = get_legal_moves(sim_state)
            if not sim_actions:
                break

            # Rollout guiado (que no sea random puro)
            # Hace cosas obvias como si puedo ganar, gano; si debo bloquear, bloqueo
            # Cpsas no tan obvias como: si soy rojo, intento odd threats - si soy amarillo, intento claim-even

            win_now = find_winning_move(
                sim_state, sim_player, sim_actions,
                simulate_move, get_winner
            )
            if win_now is not None:
                sim_action = win_now
            else:
                block_now = find_winning_move(
                    sim_state, -sim_player, sim_actions,
                    simulate_move, get_winner
                )
                if block_now is not None:
                    sim_action = block_now
                else:
                    heights = get_heights(sim_state)
                    parities = [h % 2 for h in heights]

                    if sim_player == -1:
                        odd_cols = [
                            c for c in sim_actions
                            if parities[c] == 0 and creates_odd_threat(sim_state, c, sim_player, heights)
                        ]
                        if odd_cols:
                            sim_action = ordered_moves(odd_cols, move_order)[0]
                        else:
                            sim_action = random.choice(ordered_moves(sim_actions, move_order)[:3])
                    else:
                        claim_cols = [
                            c for c in sim_actions
                            if parities[c] == 1 and claims_even(sim_state, c, sim_player, heights)
                        ]
                        if claim_cols:
                            sim_action = ordered_moves(claim_cols, move_order)[0]
                        else:
                            sim_action = random.choice(ordered_moves(sim_actions, move_order)[:3])

            sim_state = simulate_move(sim_state, sim_action, sim_player)
            sim_player = -sim_player
            depth += 1

        winner = get_winner(sim_state)
        if winner == current_player:
            result = 1
        elif winner == -current_player:
            result = -1
        else:
            result = 0

        current_node = node
        while current_node is not None:
            current_node.visits += 1
            if -current_node.player == current_player:
                current_node.wins += result
            else:
                current_node.wins += -result
            current_node = current_node.parent

    q_prime: dict[int, float] = {}
    for child in root.children:
        if child.visits > 0:
            q_prime[child.action] = child.wins / child.visits
    return q_prime

def select_from_qprime(q_prime: dict[int, float], legal_moves: list[int]) -> int:
    if not q_prime:
        return int(random.choice(legal_moves))

    vals = []
    for a in legal_moves:
        vals.append(q_prime.get(a, 0.0))

    best_idx = int(np.argmax(vals))
    return legal_moves[best_idx]

def select_final_action(s: np.ndarray, deadline: float, q_table: dict[str, float],) -> int:
    legal_moves = get_legal_moves(s)
    if not legal_moves:
        raise ValueError("No legal moves available")

    q_values = []
    for a in legal_moves:
        key = sa_key(s, a)
        q_values.append(q_table.get(key, 0.0))

    # fallback a MCTS normal si json está vacio
    if max(q_values) == min(q_values):
        current_player = get_current_player(board=s)
        best_action = quick_minimax(s, ordered_moves(legal_moves, move_order), current_player, deadline)
        if best_action is not None:
            return best_action
        return ordered_moves(legal_moves, move_order)[0]

    best_idx = int(np.argmax(q_values))
    return legal_moves[best_idx]

def state_key(board: np.ndarray) -> str:
    return board.tobytes().hex()

def sa_key(board: np.ndarray, action: int) -> str:
    return f"{state_key(board)}:{action}"