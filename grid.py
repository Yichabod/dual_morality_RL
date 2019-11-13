import random
from utils import Train, OtherMask, Switch, in_bounds
from graphics import display_grid
import numpy as np

SEED = 42
# random.seed(SEED)
# np.random.seed(SEED)

"""
To Do:
 - how to integrate reward function with transition?
 - checkRep
 - support for multiple agents?

"""


class Grid:
    '''
    Grid is the state class for the MDP, with transition and reward functions
    Provides base grid environment with which the agents interact.
    Keeps track of agent positions, others positions, and train
    Note positions are (y,x) with (0,0) in the top left corner of the grid
    '''


    def __init__(self, size, num_agents=1, random=False):
        assert isinstance(size, int)

        # available actions: stay, north, east, south, west
        # coords are (y,x), with negative being up and left
        self.all_actions =[(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)]

        self.size = size

        #all possible coordinates as a y,x tuple
        self.grid_coords = set((i,j) for i in range(size) for j in range(size))

        self.train = Train(size)
        self.other_agents = OtherMask(size)
        self.switch = Switch(size)
        self._place_grid_elements(random)
        # self._place_agent()

        #timestep horizon until end of episode
        """
        self.terminal_time = 10
        self.timestep = 0
        """
        self.terminal_state = False
        self.current_state = (self.agent_pos,self.train.pos,list(self.other_agents.positions)[0])

    def _place_grid_elements(self, random_placement=False) -> None:
        """
        places agent, train, switch, and other people
        :params:
            random_placement (boolean): whether or not the agent are placed 'randomly'
            train will always be along a border, going perpindicular to the wall. No collision

        """
        if random_placement:
            self._place_train_random()
            self._place_other_agents_random()
            self._place_switch_random()
            self._place_agent_random()
            #TODO place switch, agent. Should these be constrained to be near each other?

        else: #random
            self.train.pos = (1,0)
            self.other_agents.positions = {(1,3)}
            self.switch.pos = (0,4)
            self.agent_pos = (0,2)


    def _place_train_random(self) -> None:
        """
        randomly place train along one of the walls of the grid
        """
        orientation = np.random.choice(4)
        if orientation == 0: #against left wall
            row = np.random.choice(self.size)
            self.train.pos = (row, 0)
            self.train.velocity = (0, 1)
        elif orientation == 1: #against right wall
            row = np.random.choice(self.size)
            self.train.pos = (row, self.size-1)
            self.train.velocity = (0, -1)
        elif orientation == 2: #against top wall
            col = np.random.choice(self.size)
            self.train.pos = (0, col)
            self.train.velocity = (1, 0)
        elif orientation == 3: #against bottom wall
            col = np.random.choice(self.size)
            self.train.pos = (self.size-1, col)
            self.train.velocity = (-1, 0)

    def _place_other_agents_random(self, number=1, constrained=False) -> None:
        """
        place other agents on grid randomly
        @param: number - number of agents to place
        @param: constrained - whether or not to make sure the final agent is in path of train
        """
        placed = set()
        while number > 0:
            if constrained and number == 1: #make sure in line with train
                train_velocity = self.train.velocity
                if train_velocity == (0,1): #against left wall
                    row = self.train.pos[0]
                    col = np.random.choice(1,self.size)
                elif train_velocity == (0,-1): #against right wall
                    row = self.train.pos[0]
                    col = np.random.choice(self.size-1)
                elif train_velocity == (1,0): #against top wall
                    col = self.train.pos[1]
                    row = np.random.choice(1,self.size)
                elif train_velocity == (-1,0): #against bottom wall
                    row = self.train.pos[1]
                    col = np.random.choice(self.size-1)
                    coord = (row,col)
            else:
                coord = (np.random.choice(self.size), np.random.choice(self.size))

            if coord != self.train.pos and coord not in placed:
                number -= 1
                placed.add(coord)
        self.other_agents.positions = placed

    def _place_switch_random(self):
        """
        randomly place switch in grid without colliding with train, other agents
        """
        available_pos = set((row, col) for row in range(self.size) for col in range(self.size))\
                            -set(self.train.pos) - self.other_agents.positions
        self.switch.pos = random.sample(available_pos,1)[0]

    def _place_agent_random(self):
        """
        Randomly place agent in grid without colliding with train, other agents, or switch
        """
        try:
            available_pos = set((row, col) for row in range(self.size) for col in range(self.size))\
                            -set(self.train.pos) - self.other_agents.positions - set(self.switch.pos)
        except:
            import pdb; pdb.set_trace()
        self.agent_pos = random.sample(available_pos,1)[0]



    def legal_actions(self) -> set:
        """
        return the set of tuples representing legal actions from the current state
        """
        legal_actions = self.all_actions.copy()
        for action in self.all_actions:
            new_position_y = self.agent_pos[0]+action[0]
            new_position_x = self.agent_pos[1]+action[1]
            if not in_bounds(self.size,(new_position_y,new_position_x)):
                legal_actions.remove(action)
        return legal_actions


    def T(self, action:tuple) -> None:
        """
        Precondition: action needs to be legal, board cannot be in terminal state
        Returns None at the moment, changes grid object properties to update state
        can be changed return duplicate grid object if mutation is bad
        """

        #check not terminal state
        if self.terminal_state:
            return self.agent_pos, self.train.pos

        #check that action is legal
        new_x = self.agent_pos[0] + action[0]
        new_y = self.agent_pos[1] + action[1]
        new_agent_pos = (new_x,new_y)
        self.train.update()

        #episode ends if train leaves screen or collides
        if not self.train.on_screen:
            self.terminal_state = True

        if action not in self.legal_actions():
            new_agent_pos = self.agent_pos

        #check if switch is pushed
        if new_agent_pos == self.switch.pos:
            new_agent_pos = self.agent_pos #agent
            self.train.velocity = (self.train.velocity[1], self.train.velocity[0]) #move perpindicular
            self.switch.activated = True

        #collision detect
        if new_agent_pos == self.train.pos:
            #agent intersect train: death, terminal state
            self.train.velocity = (0,0)
            self.terminal_state = True
            pass

        if self.other_agents.positions.intersection({new_agent_pos}):
            #agent intersect other: push
            #moves both agent and other given that it will not push anyone out of bounds
            new_other_y = new_agent_pos[0] + action[0]
            new_other_x = new_agent_pos[1] + action[1]
            new_other_pos = (new_other_y,new_other_x)
            if in_bounds(self.size,new_other_pos):
                self.other_agents.push(new_agent_pos,action)
            else:
                new_agent_pos = self.agent_pos
                
        if self.other_agents.positions.intersection({self.train.pos}):
            #other intersect train: death, terminal state
            self.train.velocity = (0,0)
            self.terminal_state = True
            pass

        self.agent_pos = new_agent_pos
        self.current_state = (self.agent_pos,self.train.pos,list(self.other_agents.positions)[0])
        return self.current_state

    def R(self, action: tuple) -> int:
        """
        """
        if self.terminal_state:
            return 0

        reward = 0

        #check that action is legal
        new_x = self.agent_pos[0] + action[0]
        new_y = self.agent_pos[1] + action[1]
        new_agent_pos = (new_x,new_y)
        new_train_pos = self.train.get_next_position()

        if action not in self.legal_actions():
            new_agent_pos = self.agent_pos

        if new_agent_pos == new_train_pos:
            #agent intersect train: death
            reward -= 10

        if self.other_agents.positions.intersection({new_agent_pos}):
            #agent intersect other: push
            #moves both agent and other given that it will not push anyone out of bounds
            reward -= 1

        if self.other_agents.positions.intersection({new_train_pos}):
            #other intersect train: death, terminal state
            reward -= 100

        return reward


if __name__ == "__main__":
    # makes 5x5 test grid and chooses random action until terminal state is reached
    grid = Grid(5,random=True)
    grid.agent_pos
    display_grid(grid)
    print(grid.current_state)
    while not grid.terminal_state:
        print("")
        action =  tuple(grid.legal_actions())[0]#random.choice(tuple(grid.legal_actions()))
        grid.T(action)
        display_grid(grid)
        print(grid.current_state)
