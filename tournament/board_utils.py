import numpy as np
from typing import List

def get_winner(board: np.ndarray) -> int:
        for r in range(6):
            for c in range(4):
                p = board[r, c]
                if p != 0 and np.all(board[r, c : c + 4] == p):
                    return p
        for c in range(7):
            for r in range(3):
                p = board[r, c]
                if p != 0 and np.all(board[r : r + 4, c] == p):
                    return p
        for r in range(3):
            for c in range(4):
                p = board[r, c]
                if p != 0 and all(board[r + i, c + i] == p for i in range(4)):
                    return p
        for r in range(3):
            for c in range(3, 7):
                p = board[r, c]
                if p != 0 and all(board[r + i, c - i] == p for i in range(4)):
                    return p
        return 0

def is_terminal(board: np.ndarray) -> bool:
    return get_winner(board) != 0 or np.all(board[0] != 0)

def get_legal_moves(board: np.ndarray) -> list[int]:
    return [c for c in range(7) if board[0, c] == 0]

def simulate_move(board: np.ndarray, col: int, player: int) -> np.ndarray:
    new_board = board.copy()
    for r in range(5, -1, -1):
        if new_board[r, col] == 0:
            new_board[r, col] = player
            return new_board
    raise ValueError(f"Invalid column {col}")

def get_current_player(board: np.ndarray) -> int:
    num_red = np.count_nonzero(board == -1)
    num_yellow = np.count_nonzero(board == 1)
    return -1 if num_red == num_yellow else 1