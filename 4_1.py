'''If the initial and final states are as below and H(n): number of misplaced tiles in the current state n as compared to the goal node need to be considered as the heuristic function. You need to use Best First Search algorithm.'''



from collections import deque

# Initial and final states
initial_state = [1, 2, 5, 3, 4, 6, 7, 8, 0]
final_state = [1, 2, 3, 4, 5, 6, 7, 8, 0]

# Function to calculate the heuristic value
def heuristic(state):
    misplaced = 0
    for i in range(len(state)):
        if state[i] != final_state[i] and state[i] != 0:
            misplaced += 1
    return misplaced

# Function to generate all possible successors of a state
def generate_successors(state):
    successors = []
    zero_index = state.index(0)
    
    # Move left
    if zero_index % 3 != 0:
        new_state = state.copy()
        new_state[zero_index], new_state[zero_index - 1] = new_state[zero_index - 1], new_state[zero_index]
        successors.append(new_state)
    
    # Move right
    if zero_index % 3 != 2:
        new_state = state.copy()
        new_state[zero_index], new_state[zero_index + 1] = new_state[zero_index + 1], new_state[zero_index]
        successors.append(new_state)
    
    # Move up
    if zero_index // 3 != 0:
        new_state = state.copy()
        new_state[zero_index], new_state[zero_index - 3] = new_state[zero_index - 3], new_state[zero_index]
        successors.append(new_state)
    
    # Move down
    if zero_index // 3 != 2:
        new_state = state.copy()
        new_state[zero_index], new_state[zero_index + 3] = new_state[zero_index + 3], new_state[zero_index]
        successors.append(new_state)
    
    return successors

# Best First Search algorithm
def best_first_search():
    frontier = deque([(heuristic(initial_state), initial_state)])
    explored = set()
    
    while frontier:
        _, current_state = frontier.popleft()
        
        if current_state == final_state:
            return current_state
        
        explored.add(tuple(current_state))
        
        for successor in generate_successors(current_state):
            if tuple(successor) not in explored:
                frontier.append((heuristic(successor), successor))
                frontier = deque(sorted(frontier))
    
    return None

# Run the algorithm
result = best_first_search()
if result:
    print("Goal state found:")
    print(result)
else:
    print("Goal state not found.")