import random
from utils import Train, OtherMask, Switch, in_bounds
from graphics import display_grid
import numpy as np
import enum

SEED = 42
# random.seed(SEED)
# np.random.seed(SEED)
PEOPLE_RANGE = 10 #the number of people randomly generated goes up to this

class GridType(enum.Enum):
    AgentOnTrack = 1



class Grid:
    '''
    Grid is the state class for the MDP, with transition and reward functions
    Provides base grid environment with which the agents interact.
    Keeps track of agent positions, others positions, and train
    Note positions are (y,x) with (0,0) in the top left corner of the grid
    '''


    def __init__(self, size, random=False):
        assert isinstance(size, int)

        # available actions: stay, north, east, south, west
        # coords are (y,x), with negative being up and left
        self.all_actions =[(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)]

        self.size = size
        self.terminal_state = False

        self._place_all(random)
        self.current_state = (self.agent_pos,self.train.pos,list(self.other_agents.positions)[0])

        self.rewards_dict = {'agent hit by train': -5, 'agent pushes others':-0.5,
                            'others hit by train':-2*self.other_agents.num, 'agent push switch': -0.5,
                            'do nothing':0}

    def copy(self):
        """
        returns deep copy of grid - because grid is mutated in learning (agent performs transition and
        observes result), copy is necessary to maintain original start position
        """
        copy = Grid(self.size)
        copy.train = self.train.copy()
        copy.other_agents = self.other_agents.copy()
        copy.switch = self.switch
        copy.agent_pos = self.agent_pos
        copy.current_state = self.current_state
        return copy

    def _place_all(self, place_random, return_placement_type=True) -> None:
        """
        places agent, train, switch, and other people
        :params:
            place_random (boolean): whether or not the agent are placed 'randomly'
            train will always be along a border, going perpindicular to the wall. No collision
            Switch cannot be located in the path of the train
        """
        #all possible coordinates as a y,x tuple
        if not place_random:
            #default positions. Train, switch and other defaults found in utils.py
            self.train = Train(self.size)
            self.other_agents = OtherMask(self.size)
            self.switch = Switch(self.size)
            self.agent_pos = (0,2)
        else:
            open_grid_coords = set((i,j) for i in range(self.size) for j in range(self.size))

            train_orientation = np.random.choice(4) #random between 0-3, 0->right,1->left,2->down,3->up
            train_loc = np.random.choice(self.size) #position along starting wall
            train_map = {0:((0,1),(train_loc,0)),1:((0,-1),(train_loc,self.size-1)),
                         2:((1,0),(0, train_loc)),3:((-1,0),(self.size-1, train_loc))}
            train_vel = train_map[train_orientation][0]
            train_pos = train_map[train_orientation][1]
            self.train = Train(self.size, train_pos, train_vel)
            open_grid_coords.remove(self.train.pos)

            #sets agent position based on open coordinates
            self.agent_pos = random.sample(open_grid_coords,1)[0]
            open_grid_coords.remove(self.agent_pos)

            #sets others position based on open coordinates
            random_others_pos = random.sample(open_grid_coords,1)[0]

            #just one other for now - no NN representation exists yet
            #self.other_agents = OtherMask(self.size, positions={random_others_pos}, num=np.random.choice(PEOPLE_RANGE))
            self.other_agents = OtherMask(self.size, positions={random_others_pos}, num=1)

            open_grid_coords.remove(random_others_pos)

            #places switch so that it cannot be in path of train
            for i in range(self.size):
                train_y_coord = self.train.pos[0] + self.train.velocity[0]*i
                train_x_coord = self.train.pos[1] + self.train.velocity[1]*i
                train_coord = (train_y_coord,train_x_coord)
                if train_coord in open_grid_coords:
                    open_grid_coords.remove(train_coord)
            switch_pos = random.sample(open_grid_coords,1)[0]
            self.switch = Switch(self.size, switch_pos)


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

    def R(self, action:tuple) -> int:
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

        if self.switch.activated:
            reward += self.rewards_dict['agent push switch']
            self.switch.activated = False

        if new_agent_pos == new_train_pos:
            #agent intersect train: death
            reward += self.rewards_dict['agent hit by train']

        if self.other_agents.positions.intersection({new_agent_pos}):
            #agent intersect other: push
            #moves both agent and other given that it will not push anyone out of bounds
            reward += self.rewards_dict['agent pushes others']


        if self.other_agents.positions.intersection({new_train_pos}):
            #other intersect train: death, terminal state
            reward += self.rewards_dict['others hit by train']

        return reward


if __name__ == "__main__":
    # makes 5x5 test grid and chooses random action until terminal state is reached
    grid = Grid(5,random=True)
    display_grid(grid)
    print(grid.current_state)
    while not grid.terminal_state:
        print("")
        action =  tuple(grid.legal_actions())[0]
        grid.T(action)
        print(grid.R(action))
        display_grid(grid)
        print(grid.current_state)
