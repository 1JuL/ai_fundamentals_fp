import numpy as np
import time
from typing import Callable, Dict, List, Any

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

move_order = [3, 2, 4, 1, 5, 0, 6]
win_value = 1_000_000
window_scores = {4: win_value, 3: 5000, 2: 200, 1: 10}

# ========================= NUEVO (HELPER AMARILLO 2-PLY) =========================
def quick_minimax(board: np.ndarray, moves: list[int], player: int, deadline: float):
    # Si estamos cortos de tiempo, no hacemos nada
    if time.perf_counter() > deadline:
        return None

    legal_len = len(get_legal_moves(board))
    max_depth = 4 + (8 - legal_len) * 2

    opp = -player
    best_score = -float("inf")
    best_move = None

    moves = ordered_moves(moves, move_order)

    for a in moves:
        if time.perf_counter() > deadline:
            break

        b_after = simulate_move(board, a, player)

        # Si mi jugada gana, listo
        if get_winner(b_after) == player:
            return a

        # Ahora llamamos al nivel min (opp)
        score = min_value(b_after, 1, -float("inf"), float("inf"), deadline, player, max_depth)

        if score > best_score:
            best_score = score
            best_move = a

    return best_move

def min_value(board: np.ndarray, depth: int, alpha: float, beta: float, deadline: float, maximizing_player: int, max_depth: int) -> float:
    if time.perf_counter() > deadline:
        return heuristic(board, maximizing_player, window_scores)

    winner = get_winner(board)
    if winner == maximizing_player:
        return win_value - depth
    elif winner == -maximizing_player:
        return -win_value + depth
    if np.all(board[0] != 0):  # full board, draw
        return 0

    if depth >= max_depth:
        return heuristic(board, maximizing_player, window_scores)

    min_score = float("inf")
    legal = ordered_moves(get_legal_moves(board), move_order)
    current_player = -maximizing_player
    for a in legal:
        if time.perf_counter() > deadline:
            return heuristic(board, maximizing_player, window_scores)

        b_after = simulate_move(board, a, current_player)
        score = max_value(b_after, depth + 1, alpha, beta, deadline, maximizing_player, max_depth)
        min_score = min(min_score, score)
        beta = min(beta, min_score)
        if beta <= alpha:
            break
    return min_score

def max_value(board: np.ndarray, depth: int, alpha: float, beta: float, deadline: float, maximizing_player: int, max_depth: int) -> float:
    if time.perf_counter() > deadline:
        return heuristic(board, maximizing_player, window_scores)

    winner = get_winner(board)
    if winner == maximizing_player:
        return win_value - depth
    elif winner == -maximizing_player:
        return -win_value + depth
    if np.all(board[0] != 0):
        return 0

    if depth >= max_depth:
        return heuristic(board, maximizing_player, window_scores)

    max_score = -float("inf")
    legal = ordered_moves(get_legal_moves(board), move_order)
    current_player = maximizing_player
    for a in legal:
        if time.perf_counter() > deadline:
            return heuristic(board, maximizing_player, window_scores)

        b_after = simulate_move(board, a, current_player)
        score = min_value(b_after, depth + 1, alpha, beta, deadline, maximizing_player, max_depth)
        max_score = max(max_score, score)
        alpha = max(alpha, max_score)
        if beta <= alpha:
            break
    return max_score