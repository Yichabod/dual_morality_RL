import src.agent as agent
import src.grid as grid


#push_init_pos = {'train':(2,0),'agent':(4,1),'other1':(3,2),'switch':(0,0),'other2':(2,4),'other1num':1,'other2num':4}
#switch_init_pos = {'train':(2,0),'agent':(4,1),'other1':(0,0),'switch':(3,2),'other2':(2,4),'other1num':1,'other2num':4}

#here agent only needs to push cargo into target, which is near train. To test model free
easy1 = {'train':(1,0),'trainvel':(0,1),'cargo1':(3,2),'num1':1,'target1':(2,2),
      'switch':(0,0),'agent':(4,2),'cargo2':(2,4),'num2':2,'target2':(0,3)}
# agent should push simple cargo into target, away from train
easy2 = {'train':(1,0),'trainvel':(0,1),'cargo1':(4,1),'num1':1,'target1':(4,0),
      'switch':(0,0),'agent':(2,2),'cargo2':(2,4),'num2':2,'target2':(0,3)}
# agent should get out of the way of the train
easy3 = {'train':(1,0),'trainvel':(0,1),'cargo1':(4,1),'num1':1,'target1':(4,0),
      'switch':(0,0),'agent':(1,2),'cargo2':(2,4),'num2':2,'target2':(0,3)}
# dont_die_pos = {'train':(2,0),'trainvel':()
easy4 = {'train':(2,0),'trainvel':(0,1),'cargo1':(3,2),'num1':1,'target1':(4,0),
      'switch':(0,0),'agent':(2,2),'cargo2':(2,4),'num2':2,'target2':(0,3)}
#somewhere between 10,000 and 50,000 iterations the mc finally gets it - seems pretty hard without nn even
push3 = {'train':(1,0),'trainvel':(0,1),'cargo1':(2,3),'num1':1,'target1':(3,1),
      'switch':(4,0),'agent':(3,3),'cargo2':(2,4),'num2':2,'target2':(1,4)}

#this one takes a long time too - perhaps because reward comes too late
death1 = {'train':(0,0),'trainvel':(0,1),'cargo1':(1,2),'num1':1,'target1':(2,2),
      'switch':(4,0),'agent':(0,3),'cargo2':(2,4),'num2':2,'target2':(3,3)}

push1 = {'train':(1,0),'trainvel':(0,1),'cargo1':(2,2),'num1':1,'target1':(3,1),
  'switch':(0,0),'agent':(3,3),'cargo2':(1,4),'num2':2,'target2':(0,3)}

targets_test = {'train':(0,0),'trainvel':(0,1),'cargo1':(1,2),'num1':1,'target1':(1,3),
      'switch':(4,4),'agent':(2,1),'cargo2':(2,2),'num2':2,'target2':(3,2)}

weird1 = {'train':(4,2),'trainvel':(-1,0),'other1':(4,3),'num1':1,'target1':(0,3),
      'switch':(1,0),'agent':(3,0),'other2':(2,2),'num2':2,'target2':(2,4)}

switch = {'train':(1,0),'trainvel':(0,1),'cargo1':(2,1),'num1':1,'target1':(4,3),
      'switch':(3,3),'agent':(4,4),'cargo2':(1,2),'num2':2,'target2':(0,3)}

push4 = {'train':(2,0),'trainvel':(0,1),'cargo1':(3,2),'num1':1,'target1':(0,1),
  'switch':(4,0),'agent':(4,2),'cargo2':(3,3),'num2':2,'target2':(0,3)}

test3 = {"train": (0, 3), "trainvel": (1, 0), "cargo1": (2, 2), "target1": (0, 4), "switch": (2, 4), "agent": (2, 0), "cargo2": (3, 3), "target2": (3, 4),'num1':1,
'num2':2}

test12 = {"train": (1, 0), "trainvel": (0, 1), "cargo1": (2, 2), "target1": (3, 1), "switch": (0, 4), "agent": (3, 1), "cargo2": (1, 4), "target2": (0, 3), 'num1':1, 'num2':2}

test14 = {"train": (4, 4), "trainvel": (-1, 0), "cargo1": (1, 3), "target1": (3, 2), "switch": (4, 1), "agent": (0, 2), "cargo2": (0, 4), "target2": (0, 1), 'num1':1,
'num2':2}

test_suite = [grid.Grid(5,random=False, init_pos=test3),
            grid.Grid(5,random=False, init_pos=test12),
            grid.Grid(5,random=False, init_pos=test14)]

swit27 = {"train": (1, 0), "trainvel": (0, 1), "cargo1": (0, 1), "target1": (4, 3), "switch": (3, 3), "agent": (4, 4), "cargo2": (1, 2), "target2": (0, 3), 'num1':1, "num2":2}

if __name__ == "__main__":
  testgrid = grid.Grid(5,random=False, init_pos=death1)
  agent = agent.Agent()

  model = 'free'
  if model == 'dual':
      Q, policy = agent.mc_first_visit_control(testgrid.copy(), 1, nn_init=True,cutoff=0.4)
      agent.run_final_policy(testgrid.copy(), Q,nn_init=True,display=True)
  if model == 'free':
      agent.run_model_free_policy(testgrid.copy(), display=True)
  if model == 'based':
      Q, policy = agent.mc_first_visit_control(testgrid.copy(), iters=1000, nn_init=False)
      #display_grid(testgrid.copy())
      agent.run_final_policy(testgrid.copy(), Q,nn_init=False,display=True)
