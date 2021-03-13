"""
A simple implementation of single-player Monte Carlo Tree Search for
the purposes of our grid environment.
Our implementation is inspired by
SP-MCTS paper: https://dke.maastrichtuniversity.nl/m.winands/documents/CGSameGame.pdf
and
Luke Harold Miles, July 2019, Public Domain Dedication
https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
See also https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
"""
import math
from collections import defaultdict
from . import grid


class MCTS:
    "Monte Carlo Tree Search environment. Takes in Node and does tree search."
    def __init__(self, exploration_weight=0.5, D=1000):
        self.exploration_weight = exploration_weight
        self.D = D #large constant ensures unexplored nodes uncertain
        self.N = defaultdict(int) #number of visits for each node
        self.Q = defaultdict(int) #total reward of each node
        self.Q_sq = defaultdict(int) #total reward^2 of each node
        self.sp_UCT = defaultdict(int) #single player UCT score for each node
        self.children = dict()

    def choose(self, node):
        """
        Selects best child of node. Finds random child if no history.
        """
        if node.is_terminal():
            raise RuntimeError("choose called on terminal node {}".format(node))

        if node not self.children:
            return node.find_random_child()

        def score(n):
            if self.N[n] == 0:
                return float("-inf") #no unseen moves
            return self.sp_UCT[n]
        return max(self.children[node], key=score)

    def rollout(self, node):
        """
        Expand tree by one layer (equivalent to one iteration)
        """
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._simulate(leaf)
        self.backpropagate(path, reward)

    def _select(self, node):
        """
        Finds unexplored descendent of node.
        """
        path = [] #list of nodes from node to descendent
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                #node unexplored or is terminal (self.children[explored]=None)
                return path
            unexplored = self.children[node] - self.children.keys()
            # all the children of node that are unseen
            if unexplored: # if there are unexplored nodes at this level, return
                n = unexplored.pop()
                path.append(n)
                return path
            node = self._uct_select(node) #else use the uct heuristic to select

    def _expand(self, node):
        """
        Add children of node to children dict
        """
        if node in self.children:
            return #already added
        self.children[node] = node.find_children()

    def _simulate(self, node):
        """
        Find reward for a completed trajectory
        """
        while True:
            if node.is_terminal():
                return node.reward()
            node = node.find_random_child()

    def _backpropogate(self, path, reward):
        """
        Send reward back up to ancestors of the leaf
        """
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward
            self.Q_sq[node] += reward*reward

    def _uct_select(self, node):
        """
        Select child from node according to SP UCT equation in paper
        """
        # all children of node should already have been expanded
        assert all(n in self.children for n in self.children[node])

        log_N_parent = math.log(self.N[node]) #this is used for each child

        def sp_uct(n):
            c = self.exploration_weight
            d = self.D
            avg_reward = self.Q[n]/self.N[n]
            total_reward_sq = self.Q_sq[n]
            num_visits = self.N[n]

            UCT = avg_reward + c*math.sqrt(log_N_parent/num_visits)
            SP_mod = math.sqrt((total_reward_sq - num_visits*avg_reward**2 +D)
                                    / num_visits)
            return UCT + SP_mod

        return max(self.children[node], key=sp_uct)


class GridNode:
    """
    A representation of a game state at a discrete moment in time.
    MCTS will construct a tree of these nodes
    """
    def __init__(self, grid):
        self.grid = grid # consider initializing grids with dict so Grid
                        # class only handled within GridNode

    def find_children(self):
        if self.grid.terminal_state():
            return set()
        children = set()
        for action in self.grid.legal_actions():
            grid_temp = self.grid.copy()
            grid_temp.T(action)
            grid_temp.state_reward += grid_temp.R(action)
            children.add(grid_temp)
        return children

    def is_terminal(self):
        return self.grid.terminal_state()

    def reward(self):
        return self.grid.state_reward

    def find_random_child(self):
        for action in set(self.grid.legal_actions()):
            grid_temp = self.grid.copy()
            grid_temp.T(action)
            return grid_temp

if __name__ == "__main__":
    tree = MCTS()

    #push 1 grid init
    init1 = {'train':(1,0),'trainvel':(0,1),'other1':(2,3),'num1':1,'target1':(3,1),
            'switch':(0,0),'agent':(4,2),'other2':(1,4),'num2':2,'target2':(0,3)}

    testgrid = Grid(5,random=False, init_pos=init1)

    gridNode = GridNode(testgrid)

    for _ in range(500):
        tree.rollout(board)
