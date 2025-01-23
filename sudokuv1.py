import copy
from typing import List, Tuple
 
class SudokuBoard:
    def __init__(self, board):
        self.board = board
        self.N = 9
        self.M = 9
    def __repr__(self) -> str:
        """Provides a description of the state."""
        description = ''
        for row in range(self.N):
            description += ' '.join(str(self.board[row][col]) for col in range(self.M)) + '\n'
            #affichage de la grille avec chaque case la liste des candidats possible dans le cas il y a une case a 0
        description += '\n'
        return description
    def modif_val(self,x:int,y:int,z:int):#mise a jour de la grille
        self.board[x][y]=z
    def is_valid(self) -> bool:
        """Checks if the Sudoku board is valid."""
        # Check rows
        for row in range(self.N):
            if not self.is_valid_row(row):
                return False

        # Check columns
        for col in range(self.M):
            if not self.is_valid_column(col):
                return False

        # Check subgrids
        for row in range(0, self.N, 3):
            for col in range(0, self.M, 3):
                if not self.is_valid_subgrid(row, col):
                    return False

        return True
    def find_empty(self):
        for row in range(self.N):
            for col in range(self.N):
                if self.board[row][col] == 0:
                    return (row, col)
        return None
    def grid_full(self)->bool:
            '''la grille est pleine si toutes les cases sont remplies
            '''
            for row in range(self.N):
                for col in range(self.N):
                    if self.board[row][col] == 0:
                        return False
            return True
    def is_valid_row(self, row: int) -> bool:
        """Checks if a row in the Sudoku board is valid.
        A row is valid if it contains no duplicate non-zero numbers.
        """
        seen = set()
        for col in range(self.M):  # Iterate through each column in the row
            if self.board[row][col] != 0:  # Ignore empty cells (value 0)
                if self.board[row][col] in seen:  # Check for duplicate values
                    return False  # Invalid row if duplicate found
                seen.add(self.board[row][col])  # Add value to seen set
        return True  # Row is valid if no duplicates found
    def is_valid_column(self, col: int) -> bool:
        """Checks if a column in the Sudoku board is valid.
        A column is valid if it contains no duplicate non-zero numbers.
        """
        seen = set()
        for row in range(self.N):  # Iterate through each row in the column
            if self.board[row][col] != 0:  # Ignore empty cells (value 0)
                if self.board[row][col] in seen:  # Check for duplicate values
                    return False  # Invalid column if duplicate found
                seen.add(self.board[row][col])  # Add value to seen set
        return True  # Column is valid if no duplicates found
    
    
    def is_valid_subgrid(self, start_row: int, start_col: int) -> bool:
        """Checks if a 3x3 subgrid in the Sudoku board is valid.
        A subgrid is valid if it contains no duplicate non-zero numbers.
        """
        seen = set()
        # Iterate over the 3x3 subgrid starting from (start_row, start_col)
        for row in range(start_row, start_row + 3):
            for col in range(start_col, start_col + 3):
                if self.board[row][col] != 0:  # Ignore empty cells
                    if self.board[row][col] in seen:  # Check for duplicate values
                        return False  # Invalid subgrid if duplicate found
                    seen.add(self.board[row][col])  # Add value to seen set
        return True  # Subgrid is valid if no duplicates found
    
class SudokuSolver:
    """This class describes a formal definition of the Knight's Tour problem. 
    You should implement the remaining methods."""

    def __init__(self, board: SudokuBoard):
        """The constructor specifies the Sudoku board."""
        self.board = board
        

    def initial_state(self) -> SudokuBoard:
        """Returns the initial state"""
        return self.board
    
    def actions(self, state: SudokuBoard) -> tuple[int,int,int]:
        """Returns the actions that can be executed in the given state."""
        return self.valid_actions(state)



        
    def valid_actions(self, state: SudokuBoard) -> list[tuple[int,int,int]]:
        ''' Returns the set of valid knight moves for the given state. '''
        valid = []
        Pos = state.find_empty()
        
        if Pos is None:
            # Si aucune position n'est trouvée, retourner une liste vide
            return valid
    
        # Obtenir la liste des candidats pour la position trouvée
    
        for action in range(1,10):
            # Créer une copie du tableau et appliquer l'action
            NewStat = copy.deepcopy(state)
            NewStat.modif_val(Pos[0], Pos[1], action)
    
            # Vérifier si le nouvel état est valide
            if NewStat.is_valid():
                valid.append((action, Pos[0], Pos[1]))
    
        return valid
    def succ(self, state: SudokuBoard, action: tuple[int,int,int]) -> SudokuBoard:
        """Returns the state that results from executing the given action in the given state."""
        new_board = copy.deepcopy(state.board)
        new_board[action[1]][action[2]] = action[0]

        return SudokuBoard(new_board)

    def goal_test(self, state: SudokuBoard) -> bool:
        """Checks whether the state is a goal state."""
        return state.grid_full() and state.is_valid()

    def action_cost(self, action: Tuple[int, int]) -> int:
        """Returns the cost of a given action."""
        return 1
        #raise NotImplementedError
  

class Node:
    """A node in a search tree. A Node contains :
    - a description of the actual state for this node
    - a pointer to its parent (the node that this is a successor of) 
    - the action that got us to this state
    - the total path cost to reach the node from the root node.
    """

    def __init__(self, state: SudokuBoard, parent=None, action=None, path_cost=0):
        """Create a search tree Node, derived from a parent by an action."""
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return format(self.state)

    def expand(self, problem:SudokuSolver ) -> List["Node"]:
        """Returns the list of nodes reachable in one step from this node."""
        children = [self.child_node(problem, action)
                for action in problem.actions(self.state)]
        return children

    def child_node(self, problem: SudokuBoard, action: Tuple[int, int,int]) -> "Node":
        """Returns a node obtained by applying a given action to this node."""
        next_state = problem.succ(self.state, action)
        next_node = Node(next_state, self, action, self.path_cost + problem.action_cost(action))
        return next_node

    def solution(self) -> List[Tuple[int, int,int]]:
        """Return the sequence of actions to go from the root to this node."""
        return [node.action for node in self.path()[1:]]

    def path(self) -> List["Node"]:
        """Return a list of nodes forming the path from the root to this node."""
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    
    def equals(self, other) -> bool:
        ''' Checks if the state is the same as other. '''
        return self.state == other.state
# ______________________________________________________________________________
# Uninformed Search algorithms

def depth_first_tree_search(problem):
    """
    Search the deepest nodes in the search tree first.
    Search through the successors of a problem to find a goal.
    """
    frontier = [Node(problem.initial_state())]  # Handled as a LIFO queue (stack)
    explored = set()
    t = 0
    while frontier:
        t = t + 1
        node = frontier.pop()
        print(t)
        if problem.goal_test(node.state):
        
            return node
        frontier.extend(node.expand(problem))
    return None
    

# Example usage
board = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]

SudokuB9X9=SudokuBoard(board)  #initialize the board
Problem = SudokuSolver(SudokuB9X9) #initialize the problem
solution_node = depth_first_tree_search(Problem) #solve the problem
for node  in solution_node.path():
    print(node)#print the solution step by step