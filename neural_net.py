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

def train():
    train_xs = np.load("grids_data.npy")
    train_ys = np.load("actions_data.npy")
    action_dict = {(0, 0):0, (-1, 0):1, (0, 1):2, (1, 0):3, (0, -1):4}
    
    
    B, H, W = train_xs.shape
    C = int(np.max(train_xs))+1
    
    net = Net(C)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.03, momentum=0.9)

    
    
    # input shape b x c x h x w to net
    # so need to unsqueeze to make channel dimension of 1, could also make each agent it's own channel instead
    train_xs = torch.from_numpy(train_xs).unsqueeze(1).to(torch.long)
    onehot_train_xs = torch.zeros([B, C, H, W], dtype = torch.float32)

    onehot_train_xs.scatter_(1, train_xs, torch.ones(onehot_train_xs.shape))
    train_ys = torch.from_numpy(train_ys).to(torch.long)
    
    for epoch in range(1000):  # loop over the dataset multiple times
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

def load(C=5):
    model = Net(C)
    model.load_state_dict(torch.load('nn_model'))
    return model

def predict(model, state, C=5):
    '''
    model: pytorch model, output of load()
    input_state: HxW(5x5) numpy array
    returns 1x5(num actions) numpy array of pre-softmax probabilities for taking an action
    '''
    H, W = state.shape
    onehot_test_xs = torch.zeros([1, C, H, W])
    
    test_x = torch.from_numpy(state).unsqueeze(0).unsqueeze(1).to(torch.long)
    onehot_test_xs.scatter_(1, test_x, torch.ones(onehot_test_xs.shape))
    outputs = model(onehot_test_xs)
    
    return outputs.detach().numpy()