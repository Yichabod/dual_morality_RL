import random
import numpy as np
<<<<<<< HEAD
=======
from utils import Train, OtherMask
import random
from graphics import display_grid
>>>>>>> c44724af5493d58fa04201fdf13eb902d468cf0b


"""
To Do:
 - legal actions
 - checkRep
 - support for multiple agents?

"""


class Grid:
    '''
    Provides base grid environment with which the agents interact.
    Keeps track of agent positions
    Takes care of transitions between states

    '''

<<<<<<< HEAD
    # available actions: stay, north, east, south, west
    actions = set([(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)])


    def __init__(self, size, num_agents=1):
        assert isinstance(size, int)
        actions = set([(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)])

=======
                        

    def __init__(self, size, num_agents=1):
        assert isinstance(size, int)

        # available actions: stay, north, east, south, west
        self.all_actions = set([(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)])
        
>>>>>>> c44724af5493d58fa04201fdf13eb902d468cf0b
        self.size = size

        self.grid_coords = set((i,j) for i in range(size) for j in range(size))
        #self.wall_mask = create_wall_mask(shape)

        self.train = Train(size)
        self.other_agents = OtherMask(size)

        self._place_agent()

    def checkRep() -> None:
        """
        ensure that all states within grid are legal
        for development
        """
        pass

    def _place_agent(self) -> None:
<<<<<<< HEAD
        empty = self.grid_coords - {self.train.pos} - set(self.other_agents)
=======
        empty = self.grid_coords - {self.train.pos} - self.other_agents.get_mask_set()
>>>>>>> c44724af5493d58fa04201fdf13eb902d468cf0b
        self.agent_pos = random.choice(tuple(empty))

    def in_bounds(self,position:tuple) -> bool:
        if 0 <= position[0] < self.size and 0 <= position[1] < self.size:
            return True
        else:
            return False

    def legal_actions(self) -> set:
        """
        return the list of np arrays that are legal actions
        """
        legal_actions = self.all_actions.copy()
        for action in self.all_actions:
            new_position_y = self.agent_pos[0]+action[0]
            new_position_x = self.agent_pos[1]+action[1]
            if not self.in_bounds((new_position_y,new_position_x)):
                legal_actions -= {action}
    
        return legal_actions
<<<<<<< HEAD


=======
        
>>>>>>> c44724af5493d58fa04201fdf13eb902d468cf0b

        
    def T(self, action:tuple) -> None:
        """
        Precondition: action needs to be legal
        Returns new state, internally updates
        """
<<<<<<< HEAD
        #check that action is legal
        if action not in self.legal_actions():
            raise "Not a valid action"

        ag = self.agent_pos
        ac = action
        self.train.update
        new_agent_pos = (ag[0] + ac[0], ag[1] + ac[1])

=======

        #check that action is legal
        if action not in self.legal_actions():
            print(self.legal_actions())
            raise ValueError("Not a valid action")
        
        ag = self.agent_pos
        ac = action
        self.train.update()
        new_agent_pos = (ag[0] + ac[0], ag[1] + ac[1]) 
        
>>>>>>> c44724af5493d58fa04201fdf13eb902d468cf0b
        #collision detect
        if new_agent_pos == self.train.pos:
            #death, terminal state?
            pass
        if self.other_agents.get_mask_set().intersection({new_agent_pos}):
            #push
            new_other_y = new_agent_pos[0] + action[0]
            new_other_x = new_agent_pos[1] + action[1]
            new_other_pos = (new_other_y,new_other_x)
            if self.in_bounds(new_other_pos):
                self.other_agents.push(new_agent_pos,action)
            else:
                new_agent_pos = self.agent_pos
                
        if self.other_agents.get_mask_set().intersection({self.train.pos}):
            #death
            pass
<<<<<<< HEAD



=======
            
        self.agent_pos = new_agent_pos

    

    
        
>>>>>>> c44724af5493d58fa04201fdf13eb902d468cf0b
        #return state+action #avoid aliasing

    def R(self, state: tuple, action: tuple) -> int:
        """
        """
<<<<<<< HEAD

        #reward per state
=======
        reward = -1
        if self.agent_pos == self.train.pos:
            #death, end state
            reward = -100
        #reward per state
    

if __name__ == "__main__":
    grid = Grid(5)
    for i in range(5):
        display_grid(grid)
        print("")
        action = random.choice(tuple(grid.legal_actions()))
        grid.T(action)
>>>>>>> c44724af5493d58fa04201fdf13eb902d468cf0b
