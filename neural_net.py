from utils import generate_array
from graphics import display_grid
from grid import Grid
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self, C=5):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(C, 10, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 30, 3, padding=1)
        self.fc1 = nn.Linear(30 * 1 * 1, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 30 * 1 * 1)
        x = self.fc1(x)
        return x

def train(num_epochs=400, C=6):
    '''
    C is the number of channels in input array
    '''
    train_xs = np.load("grids_data.npy")
    train_ys = np.load("actions_data.npy")
    #
    previous_trains = train_xs[:, 0:1, :, :]
    #only keep current trains
    train_xs = train_xs[:, 1, :, :]
    #grid = Grid(5)
    #action_dict = {action:ind for ind, action in enumerate(grid.all_actions)}

    B, H, W = train_xs.shape
    #C = int(np.max(train_xs))+2

    net = Net(C)
    criterion = nn.MSELoss()#CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01)


    # input shape b x c x h x w to net
    # so need to unsqueeze to make channel dimension of 1, could also make each agent its own channel instead
    train_xs = torch.from_numpy(train_xs).unsqueeze(1).to(torch.long)
    onehot_train_xs = torch.zeros([B, C-1, H, W], dtype = torch.float32)

    onehot_train_xs.scatter_(1, train_xs, torch.ones(onehot_train_xs.shape))
    #add previous train obeservation
    onehot_train_xs = torch.cat((onehot_train_xs, torch.from_numpy(previous_trains).float()), dim=1)

    train_ys = torch.from_numpy(train_ys).float()#.to(torch.long)

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        print("epoch",epoch)
        running_loss = 0.0

        inputs, labels = onehot_train_xs, train_ys

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if epoch%100==99:
            print('Epoch:{}, loss: {:.3f}'.format(epoch + 1, running_loss))
            running_loss = 0.0
    torch.save(net.state_dict(), 'nn_model')
    print("Model saved as nn_model")

def load(C=6):
    model = Net(C)
    model.load_state_dict(torch.load('nn_model'))
    return model

def predict(model, state, C=6):
    '''
    model: pytorch model, output of load()
    input_state: 2xHxW(5x5) numpy array
    returns 1x5(num actions) numpy array of Q-values for each action
    '''
    _, H, W = state.shape
    onehot_test_xs = torch.zeros([1, C-1, H, W])

    #state[1] is current observation
    test_x = torch.from_numpy(state[1]).unsqueeze(0).unsqueeze(1).to(torch.long)
    onehot_test_xs.scatter_(1, test_x, torch.ones(onehot_test_xs.shape))

    previous_trains = state[0:1]
    onehot_test_xs = torch.cat((onehot_test_xs, torch.from_numpy(previous_trains).unsqueeze(0).float()), dim=1)
    outputs = model(onehot_test_xs)

    return outputs.detach().numpy()

if __name__ == "__main__":
    #grids = np.ones((49,2,5,5))
    #actions = np.ones((49,5))
    train()
    model = load()
    state = np.random.random([2,5,5])
    print(predict(model, state))
