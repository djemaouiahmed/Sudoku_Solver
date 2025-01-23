import copy
from typing import List, Tuple

class SudokuBoard:
    def __init__(self, board: List[List[int]]) -> None:
        """The constructor specifies the board and initializes candidates for each cell."""
        # Deep copy the board to avoid modifying the original
        sudoku = copy.deepcopy(board)
        
        # Replace each cell with its list of candidates (if the cell is empty)
        for row in range(len(board)):
            for col in range(len(board[0])):
                  # Only calculate candidates for empty cells
                if board[row][col] == 0:  # Only calculate candidates for empty cells
                    sudoku[row][col] = ListCandidat((row, col), board).candidates
                    

     
        self.board=board
        self.Candidat = sudoku
   
        self.N = len(self.board)         # Number of rows
        self.M = len(self.board[0])
        self.x_wing()      # Number of columns
        self.pairClear()               # Clear naked pairs
        self.tripleClear()             # Clear naked triples
                        # Clear x-wings
        #self.MAJGrille()


   
    def pairClear(self):
        
        self.pair_row() # Clear naked pairs in row
        self.pair_column() # Clear naked pairs in column
        self.pair_subgrid() # Clear naked pairs in subgrid
    def tripleClear(self):
        self.triple_row() # Clear naked triples in row
        self.triple_column() # Clear naked triples in column
        self.triple_subgrid() # Clear naked triples in subgrid
   
    #def MAJGrille(self):
    #    for i in range(self.N):
    #        for j in range(self.M):
    #            if self.board[i][j] == 0 and len(self.Candidat[i][j]) == 1:
    #                self.board[i][j] = self.Candidat[i][j][0]
   
   
    def __repr__(self) -> str:
        """Provides a description of the state."""
        description = ''
        for row in range(self.N):
            description += ' '.join(str(self.Candidat[row][col]) for col in range(self.M)) + '\n'
            #affichage de la grille avec chaque case la liste des candidats possible dans le cas il y a une case a 0
        description += '\n'
        return description
    def get_F_E_position(self) -> tuple[int, int] | None:
        """
        Trouve la position du noeud avec le plus petit nombre de candidats non nul.
        
        Returns:
        tuple[int, int]: La position (row, col) de la cellule, ou None si aucune position valide n'est trouvée.
        """
        min_candidates = float('inf') # Initialisation à un maximum
        position = None
 
        for row in range(self.N):
           for col in range(self.N) :
                if self.board[row][col] == 0:
                   num_candidates = len(self.getListCandidat(row, col))
                   # Mettre à jour la position si un nombre de candidats plus petit est trouvé
                   if num_candidates < min_candidates:
                      min_candidates = num_candidates
                      position = (row, col)

        return position

 
    def getListCandidat(self,x:int,y:int)->list[int]:#retourne la liste des candidats possible d'une position donner
        return self.Candidat[x][y]
    def miseAJ(self,list2:List[int],list1:List[int]):#mise a jour de la liste des candidats   
       for it in list1:
           if it in list2:
              list2.remove(it)
    
    def Clear_row(self,list:List[tuple[int,int]]):
            '''supprimer les candidate qui se retrouve dans la meme ligne qu'une paire .'''
            for i in range(self.M):
                if  i != list[0][1] and i != list[1][1] and self.board[list[0][0]][i] == 0:

                    self.miseAJ(self.Candidat[list[0][0]][i],self.Candidat[list[0][0]][list[0][1]])
    
    
    def grid_full(self)->bool:
        '''la grille est pleine si toutes les cases sont remplies
        '''
        for row in range(self.N):
            for col in range(self.N):
                if self.board[row][col] == 0:
                    return False
        return True

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
    
    def Findpair(self, list2: list[int], list1: list[int]) -> bool:
        """Checks if two candidate lists form a naked pair.
        A naked pair exists when two cells have exactly the same two candidates.
        """
        pair = set()
        # Check if both lists have exactly two elements
        if len(list1) == 2 and len(list2) == 2:
            for i in list2:  # Iterate through candidates in list2
                for j in list1:  # Iterate through candidates in list1
                    if j == i:  # Find matching elements
                        pair.add(j)  # Add matching element to pair
        return len(pair) == 2  # Return True if exactly two matches are found
    
    
    def pair_row(self):
        """Identifies naked pairs in rows and removes their candidates from other cells in the row."""
        for i in range(self.M):  # Iterate over all rows
            for j in range(self.N - 1):  # Iterate over cells in the row
                for t in range(j + 1, self.N):  # Compare the cell with subsequent cells
                    if self.board[i][j] == 0 and self.board[i][t] == 0:  # Consider only empty cells
                        if self.Findpair(self.Candidat[i][j], self.Candidat[i][t]):
                            # If a naked pair is found, clear candidates from other cells in the row
                            self.Clear_row([[i, j], [i, t]])
    
    def x_wing(self):
        """Identifies X-wings in rows and removes their candidates from other cells in the row and column."""
        for i in range(self.M):  # Iterate over all rows
            for j in range(self.N - 1):  # Iterate over cells in the row
                for t in range(j + 1, self.N):  # Compare the cell with subsequent cells
                    if self.board[i][j] == 0 and self.board[i][t] == 0:  # Consider only empty cells
                        if self.Findpair(self.Candidat[i][j], self.Candidat[i][t]):# If a naked pair is found we check if there is to other pair to have X-wing
                            for v in range(1, self.N): 
                                if v != i and self.board[v][j] == 0 and self.board[v][t] == 0 :
                                    if self.Findpair(self.Candidat[v][j],self.Candidat[i][j])and self.Findpair(self.Candidat[v][t],self.Candidat[v][t]):#if found we clear the lines and columns
                                        self.Clear_row([[i, j], [i, t]])
                                        self.Clear_row([[v, j], [v, t]])
                                        self.Clear_column([[i, j], [v, j]])
                                        self.Clear_column([[i, t], [v, t]])
    
    def pair_column(self):
        """Identifies naked pairs in columns and removes their candidates from other cells in the column."""
        for col in range(self.M):  # Iterate over all columns
            for row1 in range(self.N - 1):  # Iterate over cells in the column
                for row2 in range(row1 + 1, self.N):  # Compare the cell with subsequent cells
                    if self.board[row1][col] == 0 and self.board[row2][col] == 0:  # Consider only empty cells
                        if self.Findpair(self.Candidat[row1][col], self.Candidat[row2][col]):
                            # If a naked pair is found, clear candidates from other cells in the column
                            self.Clear_column([(row1, col), (row2, col)])
    
    
    def pair_subgrid(self):
        """Identifies naked pairs in 3x3 subgrids and removes their candidates from other cells in the subgrid."""
        # Iterate over each 3x3 subgrid
        for start_row in range(0, self.N, 3):
            for start_col in range(0, self.M, 3):
                for row1 in range(start_row, start_row + 3):
                    for col1 in range(start_col, start_col + 3):
                        for row2 in range(row1, start_row + 3):
                            for col2 in range(col1 + 1, start_col + 3):
                                if self.board[row1][col1] == 0 and self.board[row2][col2] == 0:  # Consider only empty cells
                                    if self.Findpair(self.Candidat[row1][col1], self.Candidat[row2][col2]):
                                        # If a naked pair is found, clear candidates from other cells in the subgrid
                                        self.Clear_subgrid([(row1, col1), (row2, col2)])
    
    
    def Clear_column(self, positions):
        """Removes candidates from all cells in the column except the specified positions."""
        for row in range(self.N):  # Iterate through the column
            if row not in (positions[0][0], positions[1][0]) and self.board[row][positions[0][1]] == 0:
                self.miseAJ(self.Candidat[row][positions[0][1]], self.Candidat[positions[0][0]][positions[0][1]])


    def Clear_subgrid(self, positions):
        """Removes candidates from all cells in the subgrid except the specified positions."""
        start_row = (positions[0][0] // 3) * 3
        start_col = (positions[0][1] // 3) * 3
        for row in range(start_row, start_row + 3):
            for col in range(start_col, start_col + 3):
                if (row, col) not in positions and self.board[row][col] == 0:
                    self.miseAJ(self.Candidat[row][col], self.Candidat[positions[0][0]][positions[0][1]])


    def FindTriple(self, list1: list[int], list2: list[int], list3: list[int]) -> bool:
        """
        Checks if three candidate lists form a naked triple.
        A naked triple exists when three cells share exactly three unique candidates among them.
        """
        triple = set()
        # Only consider lists with 3 or fewer candidates
        if len(list1) <= 3 and len(list2) <= 3 and len(list3) <= 3:
            for i in list1:  # Add candidates from list1 to the set
                triple.add(i)
            for i in list2:  # Add candidates from list2 to the set
                triple.add(i)
            for i in list3:  # Add candidates from list3 to the set
                triple.add(i)
        # Return True if the combined set contains exactly 3 unique candidates
        return len(triple) == 3


    def triple_row(self):
        """
        Identifies naked triples in rows and removes their candidates from other cells in the row.
        """
        for i in range(self.M):  # Iterate over all rows
            for j in range(self.N - 2):  # Iterate over cells in the row
                for k in range(j + 1, self.N - 1):  # Compare with other cells
                    for l in range(k + 1, self.N):  # Compare with a third cell
                        # Only consider empty cells
                        if self.board[i][j] == 0 and self.board[i][k] == 0 and self.board[i][l] == 0:
                            # Check if the three cells form a naked triple
                            if self.FindTriple(self.Candidat[i][j], self.Candidat[i][k], self.Candidat[i][l]):
                                # Clear candidates from other cells in the row
                                self.Clear_row_triple([(i, j), (i, k), (i, l)])
    
    
    def triple_column(self):
        """
        Identifies naked triples in columns and removes their candidates from other cells in the column.
        """
        for col in range(self.M):  # Iterate over all columns
            for row1 in range(self.N - 2):  # Iterate over cells in the column
                for row2 in range(row1 + 1, self.N - 1):  # Compare with other cells
                    for row3 in range(row2 + 1, self.N):  # Compare with a third cell
                        # Only consider empty cells
                        if self.board[row1][col] == 0 and self.board[row2][col] == 0 and self.board[row3][col] == 0:
                            # Check if the three cells form a naked triple
                            if self.FindTriple(self.Candidat[row1][col], self.Candidat[row2][col], self.Candidat[row3][col]):
                                # Clear candidates from other cells in the column
                                self.Clear_column_triple([(row1, col), (row2, col), (row3, col)])
    
        
    def triple_subgrid(self):
        """
        Identifies naked triples in 3x3 subgrids and removes their candidates from other cells in the subgrid.
        """
        # Iterate over each 3x3 subgrid
        for start_row in range(0, self.N, 3):
            for start_col in range(0, self.M, 3):
                self.process_subgrid(start_row, start_col)
    
    def process_subgrid(self, start_row, start_col):
        """
        Processes a single 3x3 subgrid to find and handle naked triples.
        """
        subgrid_cells = self.get_subgrid_cells(start_row, start_col)
        for cell1 in subgrid_cells:
            for cell2 in subgrid_cells:
                if cell1 >= cell2:  # Avoid duplicate pairs and self-pairing
                    continue
                for cell3 in subgrid_cells:
                    if cell2 >= cell3:  # Avoid duplicate triples and self-pairing
                        continue
                    self.check_and_handle_triple(cell1, cell2, cell3, subgrid_cells)
    
    def get_subgrid_cells(self, start_row, start_col):
        """
        Returns a list of cell coordinates for a given 3x3 subgrid.
        """
        cells = []
        for row in range(start_row, start_row + 3):
            for col in range(start_col, start_col + 3):
                cells.append((row, col))
        return cells
    
    def check_and_handle_triple(self, cell1, cell2, cell3, subgrid_cells):
            """
            Checks if three cells form a naked triple and clears candidates from other cells if they do.
            """
            row1, col1 = cell1
            row2, col2 = cell2
            row3, col3 = cell3
            
            # Only consider empty cells
            if self.board[row1][col1] == 0 and self.board[row2][col2] == 0 and self.board[row3][col3] == 0:
                # Check if the three cells form a naked triple
                if self.FindTriple(self.Candidat[row1][col1], self.Candidat[row2][col2], self.Candidat[row3][col3]):
                    self.clear_candidates_from_subgrid(subgrid_cells, [cell1, cell2, cell3])
    
    
    
    def clear_candidates_from_subgrid(self, subgrid_cells, triple_cells):
        """
        Clears candidates for the naked triple from other cells in the subgrid.
        """
        triple_candidates = set()
        for cell in triple_cells:
            if self.board[cell[0]][cell[1]] == 0:
               row, col = cell
   
               triple_candidates.update(self.Candidat[row][col])
       
        for cell in subgrid_cells:
            if cell not in triple_cells:
                  if self.board[cell[0]][cell[1]] == 0:
                    row, col = cell
                    self.Candidat[row][col] = [
                        candidate for candidate in self.Candidat[row][col]
                        if candidate not in triple_candidates
                ]
    def Clear_row_triple(self, positions):
        """
        Removes candidates from all cells in the row except the specified triple positions.
        """
        for i in range(self.M):  # Iterate through the row
            if (positions[0][1] != i and positions[1][1] != i and positions[2][1] != i
                    and self.board[positions[0][0]][i] == 0):  # Skip the triple and filled cells
                # Remove candidates belonging to the naked triple from this cell
                self.miseAJ(self.Candidat[positions[0][0]][i], self.Candidat[positions[0][0]][positions[0][1]])
    
    
    def Clear_column_triple(self, positions):
        """
        Removes candidates from all cells in the column except the specified triple positions.
        """
        for row in range(self.N):  # Iterate through the column
            if (positions[0][0] != row and positions[1][0] != row and positions[2][0] != row
                    and self.board[row][positions[0][1]] == 0):  # Skip the triple and filled cells
                # Remove candidates belonging to the naked triple from this cell
                self.miseAJ(self.Candidat[row][positions[0][1]], self.Candidat[positions[0][0]][positions[0][1]])
    
    
    def Clear_subgrid_triple(self, positions):
        """
        Removes candidates from all cells in the subgrid except the specified triple positions.
        """
        # Determine the starting position of the subgrid
        start_row = (positions[0][0] // 3) * 3
        start_col = (positions[0][1] // 3) * 3
        # Iterate over all cells in the subgrid
        for row in range(start_row, start_row + 3):
            for col in range(start_col, start_col + 3):
                if (row, col) not in positions and self.board[row][col] == 0:  # Skip the triple and filled cells
                    # Remove candidates belonging to the naked triple from this cell
                    self.miseAJ(self.Candidat[row][col], self.Candidat[positions[0][0]][positions[0][1]])
    
    





class ListCandidat:
    def __init__(self, Position: tuple[int, int], stat: list[list[int]]):#initalisation de la list des candidats 
        self.candidates = self.findCandidat(Position, stat)
    
    def findCandidat(self, Position: tuple[int, int], stat: list[list[int]]) -> list[int]:
        board = copy.deepcopy(stat)  # Copy the board
        candidat = [1, 2, 3, 4, 5, 6, 7, 8, 9]

        # Remove candidates based on the row
        for row in range(len(board)):
            if board[row][Position[1]] in candidat:
                candidat.remove(board[row][Position[1]])
        
        # Remove candidates based on the column
        for col in range(len(board[0])):
            if board[Position[0]][col] in candidat:
                candidat.remove(board[Position[0]][col])
        
        # Determine the top-left corner of the subgrid
        start_row = (Position[0] // 3) * 3
        start_col = (Position[1] // 3) * 3
        
        # Remove candidates based on the subgrid
        for row in range(start_row, start_row + 3):
            for col in range(start_col, start_col + 3):
                if board[row][col] in candidat:
                    candidat.remove(board[row][col])
        
        return candidat

   
      
   
    

      

  



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
        Pos = state.get_F_E_position()
        
        if Pos is None:
            # Si aucune position n'est trouvée, retourner une liste vide
            return valid
    
        # Obtenir la liste des candidats pour la position trouvée
        candidates = state.getListCandidat(Pos[0], Pos[1])
    
        for action in candidates:
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


# ______________________________________________________________________________


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
        
        if problem.goal_test(node.state):
            print(t)
            return node
        if node.state in explored:
            print("Cycle detected")
            continue
        explored.add(node.state)
        frontier.extend(node.expand(problem))
    return None
    

# ______________________________________________________________________________
# Solve
board = [
    [0, 7, 0, 0, 0, 5, 4, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 9, 1],
    [0, 3, 0, 1, 0, 0, 0, 5, 0],
    [0, 2, 0, 0, 0, 4, 6, 0, 0],
    [3, 0, 0, 0, 8, 0, 5, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [6, 0, 0, 0, 0, 2, 3, 0, 0],
    [8, 0, 0, 0, 0, 0, 9, 0, 6],
    [9, 0, 0, 0, 0, 1, 0, 0, 0]
]
SudokuB9X9=SudokuBoard(board)  #initialize the board
Problem = SudokuSolver(SudokuB9X9) #initialize the problem
solution_node = depth_first_tree_search(Problem) #solve the problem
for node  in solution_node.path():
        print(node)#print the solution step by step
