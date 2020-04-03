import random
from utils import Train, OtherMask, Switch, in_bounds
from graphics import display_grid
import numpy as np


class Grid:
    '''
    Grid is the state class for the MDP, with transition and reward functions
    Provides base grid environment with which the agents interact.
    Keeps track of agent positions, others positions, and train
    Note positions are (y,x) with (0,0) in the top left corner of the grid
    Args:
        size - size of grid
        random - whether placement of agents is random
        init_pos - initial positions for the elements in the grid. Should be in
            this format: {'train':(1,1),'agent':(2,2),'other1':(2,1),'switch':(4,1),'other2':(2,0),'other1num':3,...}
    '''


    def __init__(self, size=5, random=False, init_pos={}):
        assert isinstance(size, int)

        # available actions: stay, north, east, south, west
        # coords are (y,x), with negative being up and left
        self.all_actions =[(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)]

        self.size = size
        self.terminal_state = False
        self.step = 1
        self.alive = True

        self._place_all(random, init_pos)
        # self.current_state = (self.agent_pos,self.train.pos,list(self.other_agents.positions)[0])
        self.current_state = (self.agent_pos,self.train.pos)+tuple(self.other_agents.positions)

        self.rewards_dict = {'agent hit by train': -10, 'agent pushes others':0,
                            'others hit by train':-1, 'agent push switch': 0, 
                            'others on target': 1, 'do nothing':0}

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

    def _place_all(self, place_random, init_pos) -> None:
        """
        places agent, train, switch, and other people
        :params:
            place_random (boolean): whether or not the agent are placed 'randomly'
            train will always be along a border, going perpindicular to the wall. No collision
            Switch cannot be located in the path of the train
        """
        #all possible coordinates as a y,x tuple
        if not place_random or len(init_pos) > 0:
            #default positions. Train, switch and other defaults found in utils.py
            if len(init_pos) == 0:
                self.train = Train(self.size)
                self.other_agents = OtherMask(self.size)
                self.switch = Switch(self.size)
                self.agent_pos = (0,2)
            else:
                self.train = Train(self.size,pos=init_pos['train'],velocity=init_pos['trainvel'])
                self.agent_pos = init_pos['agent']
                self.switch = Switch(self.size,pos=init_pos['switch'])
                others_pos = [init_pos['other1'],]
                targets = [init_pos['target1']]
                num = [init_pos['num1'],]
                if 'num2' in init_pos:
                    others_pos.append(init_pos['other2'])
                    num.append(init_pos['num2'])
                    targets.append(init_pos['target2'])
                self.other_agents = OtherMask(self.size, positions=others_pos, num=num, targets=targets)
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

            #sets others position and their targets based on open coordinates
            random_others_pos = list(random.sample(open_grid_coords, 2))
            open_grid_coords -= set(random_others_pos)
            random_targets_pos = list(random.sample(open_grid_coords, 2))
            open_grid_coords -= set(random_targets_pos)

            #feed in pos, num and target, with index 0 corresponding to first object, and idx 1 corresponding to second
            self.other_agents = OtherMask(self.size, positions=random_others_pos, num=[1,2], targets = random_targets_pos)

            open_grid_coords -= set(random_others_pos)

            #switch can be in any open position remaining
            switch_pos = random.sample(open_grid_coords,1)[0]
            self.switch = Switch(self.size, switch_pos)

            """
            #places switch so that it cannot be in path of train
            for i in range(self.size):
                train_y_coord = self.train.pos[0] + self.train.velocity[0]*i
                train_x_coord = self.train.pos[1] + self.train.velocity[1]*i
                train_coord = (train_y_coord,train_x_coord)
                if train_coord in open_grid_coords:
                    open_grid_coords.remove(train_coord)
            switch_pos = random.sample(open_grid_coords,1)[0]
            self.switch = Switch(self.size, switch_pos)
            """


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

        if not self.alive:
            action = (0,0)

        #check that action is legal
        new_x = self.agent_pos[0] + action[0]
        new_y = self.agent_pos[1] + action[1]
        new_agent_pos = (new_x,new_y)

        if action not in self.legal_actions():
            new_agent_pos = self.agent_pos

        #check if switch is pushed
        if new_agent_pos == self.switch.pos:
            new_agent_pos = self.agent_pos #agent
            self.train.velocity = (self.train.velocity[1], self.train.velocity[0]) #move perpindicular
        old_train_pos = self.train.pos
        self.train.update() #update train AFTER switch is hit
        
        #episode ends if train leaves screen or collides
        if self.step==5:
            self.terminal_state = True

        if new_agent_pos in self.other_agents.positions:
            #agent intersect other: push
            #moves both agent and other given that it will not push anyone out of bounds
            new_other_y = new_agent_pos[0] + action[0]
            new_other_x = new_agent_pos[1] + action[1]
            new_other_pos = (new_other_y,new_other_x)
            #no pushing allowed if crash has already occured
            train_stopped = new_agent_pos == self.train.pos and self.train.velocity == (0,0)
            #no pushing if another object or switch is in next location
            pos_open = new_other_pos not in self.other_agents.positions and new_other_pos != self.switch.pos 
            
            if in_bounds(self.size,new_other_pos) and pos_open and not train_stopped:
                self.other_agents.push(new_agent_pos,action)
            else:
                new_agent_pos = self.agent_pos

        #collision detect
        if new_agent_pos == self.train.pos:
            #agent intersect train: death, terminal state
            self.train.velocity = (0,0)
            self.alive = False

        if (self.agent_pos == self.train.pos) and (new_agent_pos == old_train_pos):
            #agent should not be able to pass through train
            self.train.velocity = (0,0)
            new_agent_pos = self.agent_pos
            self.alive = False

        if self.train.pos in self.other_agents.positions:
            #other intersect train: death, terminal state
            self.train.velocity = (0,0)

        self.agent_pos = new_agent_pos
        self.step += 1
        self.current_state = (self.agent_pos,self.train.pos,list(self.other_agents.positions)[0])
        return self.current_state

    def R(self, action:tuple) -> int:
        """
        """

        reward = 0

        if self.terminal_state:
            return reward

        #check that action is legal
        new_x = self.agent_pos[0] + action[0]
        new_y = self.agent_pos[1] + action[1]
        new_agent_pos = (new_x,new_y)
        new_train_pos = self.train.get_next_position(self.train.velocity)

        if action not in self.legal_actions():
            new_agent_pos = self.agent_pos

        if new_agent_pos == self.switch.pos:
            reward += self.rewards_dict['agent push switch']
            new_agent_pos = self.agent_pos
            new_train_pos = self.train.get_next_position((self.train.velocity[1], self.train.velocity[0]))

        train_active = self.train.velocity != (0,0)

        if new_agent_pos == new_train_pos and train_active:
            #agent intersect train: death
            reward += self.rewards_dict['agent hit by train']

        elif (self.agent_pos == new_train_pos) and (new_agent_pos == self.train.pos) and train_active:
            #agent intersect train: death
            reward += self.rewards_dict['agent hit by train']

        new_agent_mask = {}
        for other_pos in self.other_agents.mask.keys():
            if new_agent_pos == other_pos:
                reward += self.rewards_dict['agent pushes others']
                new_other_y = new_agent_pos[0] + action[0]
                new_other_x = new_agent_pos[1] + action[1]
                new_other_pos = (new_other_y,new_other_x)
                train_stopped = new_agent_pos == self.train.pos and self.train.velocity == (0,0)
                pos_open = new_other_pos not in self.other_agents.positions and new_other_pos != self.switch.pos
                if not in_bounds(self.size,new_other_pos) or not pos_open or train_stopped:
                    new_other_pos = other_pos
                new_agent_mask[new_other_pos] = self.other_agents.mask[other_pos]
            new_agent_mask[other_pos] = self.other_agents.mask[other_pos]
                
        #other intersect train: death, terminal state
        for pos in new_agent_mask.keys():
            if pos == new_train_pos and train_active:
                reward += self.rewards_dict['others hit by train'] * new_agent_mask[pos].get_num()
            if self.step == 5 and pos == new_agent_mask[pos].get_target() and pos != self.train.pos:
                reward += self.rewards_dict['others on target'] * new_agent_mask[pos].get_num()

        return reward

wasd_dict = {'w':(-1,0),'a':(0,-1),'s':(1,0),'d':(0,1),' ':(0,0)}
if __name__ == "__main__":
    # makes 5x5 test grid and chooses random action until terminal state is reached
    
    #push 1 grid init
    init1 = {'train':(1,0),'trainvel':(0,1),'other1':(2,3),'num1':1,'target1':(3,1),
            'switch':(0,0),'agent':(4,2),'other2':(1,4),'num2':2,'target2':(0,3)}

    #push 3 grid init
    init3 = {'train':(1,0),'trainvel':(0,1),'other1':(2,3),'num1':1,'target1':(3,1),
            'switch':(4,0),'agent':(3,3),'other2':(2,4),'num2':2,'target2':(1,4)}

    grid = Grid(5,random=True,init_pos={})
    display_grid(grid)
    print(grid.current_state)
    reward = 0
    while not grid.terminal_state:
         #wasd-space (space to stay in place)
        i = input('next step: ')
        action = wasd_dict[i]

        print("")
        reward += grid.R(action)

        grid.T(action)
        display_grid(grid)
        print(grid.current_state)
    print(reward)
