
# Sudoku Solver

## Overview
This project is a **Python-based Sudoku Solver** that employs a combination of logical deduction and a backtracking search algorithm to solve Sudoku puzzles efficiently. By systematically narrowing down possibilities and applying advanced techniques, the solver can handle puzzles ranging from simple to extremely challenging.

## Features
- **Logical Deduction**: Implements advanced strategies to simplify puzzles by eliminating impossible candidates.
- **Search Algorithm**: Employs a **backtracking search** to explore and finalize solutions when logical techniques alone cannot complete the puzzle.
- **Original Puzzle Preservation**: Keeps the input board unchanged while solving, making it easy to compare solutions.
- **Customizable Framework**: Easily extendable to include new solving techniques or optimizations.

## Algorithms Used

### 1. Logical Deduction
The solver applies logical strategies to eliminate candidates and deduce values:
- **Single Candidate Elimination**: Identifies cells with only one possible value and updates their neighbors.
- **Naked Pair Elimination**: Removes pairs of candidates shared by two cells in a row, column, or subgrid from other candidates.
- **Naked Triple Elimination**: Extends the logic of naked pairs to groups of three cells.
- **X-Wing**: Analyzes rows and columns for repeating patterns to eliminate candidates based on cross-line constraints.

### 2. Backtracking Search
When logical deduction cannot progress further, the solver uses a **backtracking search algorithm**:
- Selects an empty cell with the fewest remaining candidates (minimum remaining values heuristic).
- Tries each candidate recursively while ensuring the puzzle's constraints are satisfied.
- Backtracks if a conflict is encountered, ensuring no invalid states are reached.

### 3. Constraint Propagation
- Updates candidates dynamically as values are deduced.
- Ensures all constraints (row, column, subgrid) are respected throughout the solving process.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/sudoku-solver.git
   cd sudoku-solver
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Extract files if downloaded in `.7z` format:
   ```bash
   7z x Sudoku_project.7z
   ```

## Usage
1. Initialize the solver with a 9x9 Sudoku puzzle (as a 2D list).
2. The solver will:
   - Apply logical techniques (e.g., single candidate elimination, naked pairs, etc.).
   - Use backtracking search if required to find the solution.
3. View the solved board or debug logs for insights into the solving process.

## Example Workflow
- Input a Sudoku puzzle as a 9x9 grid.
- The solver first applies logical techniques to simplify the puzzle.
- If necessary, it switches to backtracking search to finalize the solution.
- Output: A fully solved Sudoku board, along with optional logs showing the solving steps.

## Project Structure
- **`sudoku_solver.py`**: Core implementation of the solver.
- **`examples/`**: Includes sample puzzles for testing.
- **`tests/`**: Unit tests to validate the solver's accuracy and efficiency.

## Example Results
- For simple puzzles, the solver completes the solution using only logical techniques.
- For harder puzzles, a combination of logic and backtracking ensures a solution is found.

## Contributing
Contributions are welcome! If you’d like to add new solving techniques, optimize existing algorithms, or improve documentation, feel free to submit an issue or a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
