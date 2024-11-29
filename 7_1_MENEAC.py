import random
from collections import defaultdict

class MENACE:
    def __init__(self):
        self.matchboxes = defaultdict(lambda: defaultdict(int))  # Matchboxes for board states
        self.history = []  # To track moves during a game
        self.initial_beads = 4  # Initial number of beads for each move

    def initialize_state(self, board_state):
        """
        Initialize beads for a new board state.
        :param board_state: Tuple representing the board state
        """
        if board_state not in self.matchboxes:
            for move in range(9):
                if board_state[move] == ' ':
                    self.matchboxes[board_state][move] = self.initial_beads

    def select_move(self, board_state):
        """
        Select a move based on the current board state.
        :param board_state: Tuple representing the board state
        :return: Move index (0-8)
        """
        self.initialize_state(board_state)
        moves = self.matchboxes[board_state]
        total_beads = sum(moves.values())
        if total_beads == 0:
            return random.choice([i for i, x in enumerate(board_state) if x == ' '])
        move_probs = [moves[move] / total_beads for move in range(9)]
        move = random.choices(range(9), weights=move_probs, k=1)[0]
        self.history.append((board_state, move))
        return move

    def update(self, result):
        """
        Update bead counts based on the game result.
        :param result: 1 for win, -1 for loss, 0 for draw
        """
        for board_state, move in self.history:
            if result == 1:  # Reward for winning
                self.matchboxes[board_state][move] += 3
            elif result == -1:  # Penalty for losing
                self.matchboxes[board_state][move] = max(1, self.matchboxes[board_state][move] - 1)
        self.history = []

    def print_matchboxes(self):
        """
        Print the matchboxes for debugging.
        """
        for state, moves in self.matchboxes.items():
            print(f"State: {''.join(state)}")
            print("Moves:", moves)

def check_winner(board):
    """
    Check if there is a winner in the Tic-Tac-Toe board.
    :param board: List representing the board
    :return: 'X', 'O', or None for no winner
    """
    winning_positions = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),  # Rows
        (0, 3, 6), (1, 4, 7), (2, 5, 8),  # Columns
        (0, 4, 8), (2, 4, 6)  # Diagonals
    ]
    for a, b, c in winning_positions:
        if board[a] == board[b] == board[c] and board[a] != ' ':
            return board[a]
    return None

def is_draw(board):
    """
    Check if the board state is a draw.
    :param board: List representing the board
    :return: True if draw, False otherwise
    """
    return ' ' not in board

def play_game():
    """
    Play a game of Tic-Tac-Toe with MENACE as one of the players.
    """
    menace = MENACE()
    for _ in range(100):  # Train MENACE through 100 games
        board = [' '] * 9
        current_player = 'X'
        menace_player = 'X'  # MENACE plays as 'X'

        while True:
            if current_player == menace_player:
                state = tuple(board)
                move = menace.select_move(state)
            else:
                available_moves = [i for i, x in enumerate(board) if x == ' ']
                move = random.choice(available_moves)

            board[move] = current_player
            winner = check_winner(board)
            if winner:
                menace.update(1 if winner == menace_player else -1)
                break
            elif is_draw(board):
                menace.update(0)
                break

            current_player = 'O' if current_player == 'X' else 'X'

    menace.print_matchboxes()

if __name__ == "__main__":
    play_game()
