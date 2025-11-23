import numpy as np
import random
import math

ROWS = 6
COLS = 7

# HELPERS ALLIS + MINIMAX
#  Implementa todas las funciones necesarias para:
#   - paridad y alturas (Allis),
#   - odd threats / claim-even / follow-up,
#   - heurística minimax por ventanas,
#   - detección de bloqueos y amenazas útiles.

# En pocas palabras, los helpers para implementar lo bueno de las otras policies


def ordered_moves(moves: list[int], move_order: list[int]) -> list[int]:
    return sorted(moves, key=lambda c: move_order.index(c))


def get_heights(board: np.ndarray) -> list[int]:
    heights = [0] * COLS
    for c in range(COLS):
        for r in range(ROWS):
            if board[r, c] != 0:
                heights[c] = ROWS - r
                break
    return heights


def creates_odd_threat(board: np.ndarray, col: int, player: int, heights: list[int]) -> bool:
    new_height = heights[col] + 1
    if new_height > ROWS:
        return False
    if new_height % 2 == 1:
        return potential_threat(board, col, new_height, player)
    return False


def claims_even(board: np.ndarray, col: int, player: int, heights: list[int]) -> bool:
    new_height = heights[col] + 1
    if new_height > ROWS:
        return False
    if new_height % 2 == 0:
        return potential_threat(board, col, new_height, player) or base_inverse_check(heights[col])
    return False


def base_inverse_check(current_height: int) -> bool:
    return current_height == 0 or (current_height % 2 == 1)


def is_followup_threat(board: np.ndarray, col: int, opp: int) -> bool:
    heights = get_heights(board)
    h = heights[col]
    if h >= ROWS:
        return False
    return potential_threat(board, col, h + 1, opp)


def opponent_can_respond(board: np.ndarray, col: int, player: int,
                         get_legal_moves_fn, simulate_move_fn, get_winner_fn) -> bool:
    opp = -player
    new_board = simulate_move_fn(board, col, player)
    opp_moves = get_legal_moves_fn(new_board)
    for oc in opp_moves:
        ob = simulate_move_fn(new_board, oc, opp)
        if get_winner_fn(ob) != player:
            return True
    return False


def is_useful_threat(board: np.ndarray, col: int, player: int,
                     heights: list[int], parities: list[int]) -> bool:
    new_height = heights[col] + 1
    if new_height > ROWS:
        return False

    r = ROWS - new_height
    if not (0 <= r < ROWS):
        return False

    above_r = r - 1
    if above_r >= 0 and board[above_r, col] == -player:
        return False

    global_parity = sum(ROWS - h for h in heights) % 2
    return (new_height % 2 == 1 if player == -1 else new_height % 2 == 0) \
           or global_parity == (1 if player == -1 else 0)


def potential_threat(board: np.ndarray, col: int, height: int, player: int) -> bool:
    if height < 1 or height > ROWS:
        return False
    r = ROWS - height
    if not (0 <= r < ROWS):
        return False

    count_h = 1 + sum(
        1 for dc in (-1, 1)
        if 0 <= col + dc < COLS and board[r, col + dc] == player
    )
    count_v = 1 + (1 if r + 1 < ROWS and board[r + 1, col] == player else 0)
    count_d1 = 1 + sum(
        1 for d in (-1, 1)
        if 0 <= col + d < COLS and 0 <= r + d < ROWS and board[r + d, col + d] == player
    )
    count_d2 = 1 + sum(
        1 for d in (-1, 1)
        if 0 <= col + d < COLS and 0 <= r - d < ROWS and board[r - d, col + d] == player
    )
    return max(count_h, count_v, count_d1, count_d2) >= 2


def heuristic(board: np.ndarray, player: int, window_scores: dict[int, float]) -> int:
    return count_windows(board, player, window_scores) - count_windows(board, -player, window_scores)


def count_windows(board: np.ndarray, player: int, window_scores: dict[int, float]) -> int:
    score = 0
    for r in range(ROWS):
        for c in range(COLS - 3):
            window = board[r, c:c+4]
            score += window_score(window, player, window_scores)
    for c in range(COLS):
        for r in range(ROWS - 3):
            window = board[r:r+4, c]
            score += window_score(window, player, window_scores)
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            window = np.array([board[r+i, c+i] for i in range(4)])
            score += window_score(window, player, window_scores)
    for r in range(ROWS - 3):
        for c in range(3, COLS):
            window = np.array([board[r+i, c-i] for i in range(4)])
            score += window_score(window, player, window_scores)
    return score


def window_score(window: np.ndarray, player: int, window_scores: dict[int, float]) -> int:
    p_count = np.count_nonzero(window == player)
    o_count = np.count_nonzero(window == -player)
    if o_count > 0:
        return 0
    return window_scores.get(p_count, 0)


def find_winning_move(board: np.ndarray, player: int, legal_moves: list[int],
                      simulate_move_fn, get_winner_fn):
    for c in legal_moves:
        b2 = simulate_move_fn(board, c, player)
        if get_winner_fn(b2) == player:
            return c
    return None


def is_block_move(board: np.ndarray, action: int, player: int,
                  get_legal_moves_fn, simulate_move_fn, get_winner_fn) -> bool:
    opp = -player
    legal = get_legal_moves_fn(board)
    for c in legal:
        b2 = simulate_move_fn(board, c, opp)
        if get_winner_fn(b2) == opp:
            return action == c
    return False


def creates_useful_threat(board: np.ndarray, action: int, player: int,
                          simulate_move_fn, window_scores: dict[int, float]) -> bool:
    b2 = simulate_move_fn(board, action, player)
    return count_windows(b2, player, window_scores) > count_windows(board, player, window_scores)
