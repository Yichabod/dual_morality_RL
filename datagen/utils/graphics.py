import numpy as np


def display_grid(mdp, action=None):

    """
    Takes ina Grid mdp environment and an agent, with optional action to
    show which direction next move should be in
    Displays main agent(◉), other agents(numbers), switch(S) and train (T)
    in the grid of given dimensions
    Call the displayGrid function right after next action is generated
    but before it moves
    To Do: add support for collision
    """
    dims = (mdp.size,mdp.size) #tuple eg (11,11)
    grid = np.full(dims, "_", dtype=str) #np has nice display built in
    others_dict = mdp.other_agents.mask

    for other in others_dict:
        grid[other[0],other[1]] = str(others_dict[other])

    #if type(action) == np.ndarray:
        #next_x, next_y = action+agent.state
        #grid[next_x,next_y] = "N"

    grid[mdp.agent_pos[0],mdp.agent_pos[1]] = "◉" #where the agent is
    grid[mdp.switch.pos[0], mdp.switch.pos[1]] = "S"
    if mdp.train.on_screen == True:
        # if agent is killed by train, X marks collision
        if mdp.train.pos == mdp.agent_pos:
            grid[mdp.train.pos[0],mdp.train.pos[1]] = "X"
        # if other is killed by train, X marks collision
        elif set(others_dict.keys()).intersection({mdp.train.pos}):
            grid[mdp.train.pos[0],mdp.train.pos[1]] = "x"
        else:
            grid[mdp.train.pos[0],mdp.train.pos[1]] = "T"

    print(grid)
    return grid
