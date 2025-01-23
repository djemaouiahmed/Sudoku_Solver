import random

def generate_full_board():
    """
    Generates a fully solved 9x9 Sudoku board using backtracking.
    """
    board = [[0] * 9 for _ in range(9)]

    def is_valid(board, row, col, num):
        for x in range(9):
            if board[row][x] == num or board[x][col] == num:
                return False
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(start_row, start_row + 3):
            for j in range(start_col, start_col + 3):
                if board[i][j] == num:
                    return False
        return True

    def solve(board):
        for row in range(9):
            for col in range(9):
                if board[row][col] == 0:
                    for num in range(1, 10):
                        if is_valid(board, row, col, num):
                            board[row][col] = num
                            if solve(board):
                                return True
                            board[row][col] = 0
                    return False
        return True

    solve(board)
    return board

def remove_values(board, difficulty):
    """
    Removes values from a solved board to create a puzzle.

    Args:
        board (list): 9x9 solved Sudoku board.
        difficulty (str): Difficulty level ('easy', 'medium', 'hard').

    Returns:
        list: Sudoku puzzle with values removed.
    """
    removal_count = {
        "easy": 20,
        "medium": 40,
        "hard": 60,
    }.get(difficulty, 20)
    puzzle = [row[:] for row in board]
    for _ in range(removal_count):
        row, col = random.randint(0, 8), random.randint(0, 8)
        while puzzle[row][col] == 0:
            row, col = random.randint(0, 8), random.randint(0, 8)
        puzzle[row][col] = 0
    return puzzle

def save_puzzles_to_file(puzzles, file_path):
    """
    Saves Sudoku puzzles to a file.

    Args:
        puzzles (list): List of Sudoku puzzles.
        file_path (str): Path to the file to save puzzles.
    """
    with open(file_path, "w") as f:
        for i, puzzle in enumerate(puzzles):
            f.write(f"# Puzzle {i + 1}\n")
            for row in puzzle:
                f.write(" ".join(map(str, row)) + "\n")
            f.write("\n")

def generate_and_save_puzzles(num_puzzles, difficulty, file_path):
    """
    Generates and saves Sudoku puzzles to a file.

    Args:
        num_puzzles (int): Number of puzzles to generate.
        difficulty (str): Difficulty level ('easy', 'medium', 'hard').
        file_path (str): Path to the file to save puzzles.
    """
    puzzles = [remove_values(generate_full_board(), difficulty) for _ in range(num_puzzles)]
    save_puzzles_to_file(puzzles, file_path)
    print(f"Generated {num_puzzles} {difficulty} puzzles and saved to {file_path}")

# Example: Generate 100 medium-difficulty puzzles and save to a file
generate_and_save_puzzles(100, "hard", "sudoku_puzzles.txt")
