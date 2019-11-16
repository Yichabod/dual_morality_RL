import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import generate_array
import utils
from graphics import display_grid
from grid import Grid
import numpy as np



ACTION_DICT = {(0, 0):0, (-1, 0):1, (0, 1):2, (1, 0):3, (0, -1):4}



train_xs = np.load("grids_data.npy")
train_ys = np.load("actions_data.npy")

print(train_xs.shape, train_ys.shape)
C = int(np.max(train_xs))+1 #+1 since 0 means no channels

class Neuralnet(nn.Module):
    """
    Expected input = (Batch, Channels, Height, Width), in the test case (~900, 5, 5, 5)
    """
    def __init__(self):
        super(Neuralnet, self).__init__()
        self.conv1 = nn.Conv2d(C, 10,5,padding=2)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(10,30,3,padding=1)
        self.fc1 = nn.Linear(30 * 1 * 1, 5)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 30 * 1 * 1)
        x = self.fc1(x)
        return x

net = Neuralnet()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# input shape b x c x h x w to net
# so need to unsqueeze to make channel dimension of 1, could also make each agent its own channel instead
B, H, W = train_xs.shape
train_xs[0]
train_xs = torch.from_numpy(train_xs).unsqueeze(1) #dims: (num_examples, 1, grid.size, grid.size)
train_xs.shape
onehot_train_xs = torch.zeros(B, C, H, W)

print(train_xs.shape, onehot_train_xs.shape)
onehot_train_xs.scatter_(1, train_xs, torch.ones(onehot_train_xs.shape)) #take the arrays and one-hot them
print(onehot_train_xs[0])
train_ys = torch.from_numpy(train_ys)

def train():
    for epoch in range(300):  # loop over the dataset multiple times

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
        if epoch%10==9:
            print('Epoch:{}, loss: {:.3f}'.format(epoch + 1, running_loss))
            running_loss = 0.0

def predict(input_state):
    '''
    input_state: HxW (5x5) numpy array
    returns 1x5(num actions) numpy array of pre-softmax probabilities of actions
    '''
    pass
    
onehot_train_xs = torch.zeros(B, C, H, W)
test = np.array([[0, 2, 0, 0, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 3],
        [0, 0, 0, 4, 0]])
onehot_train_xs.scatter_(1, test, torch.ones(onehot_train_xs.shape))

test1 = torch.from_numpy(np.array([[[[0, 2, 0, 0, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 3],
        [0, 0, 0, 4, 0]]]]))
test1.shape
outputs = net(test1)
