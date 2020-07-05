import numpy as np
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

cuda = True if torch.cuda.is_available() else False
NUM_TARGETS = 2
CHANNELS = 9

NN_FILE = 'nn_model'

class Net(nn.Module):
    def __init__(self, C=CHANNELS, dropout_p=0.2):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(C, 100, 3, padding=1) # mixing channels a bad thing? Look at other deep gridworld architectures
        self.conv2 = nn.Conv2d(100, 100, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(100, 100, 3, padding=1)
        self.conv4 = nn.Conv2d(100, 100, 3, padding=1)
        # self.conv3_1 = nn.Conv2d(100, 100, 3, padding=1)

        # self.conv3_2 = nn.Conv2d(100, 100, 3, padding=1)
        # self.conv3_3 = nn.Conv2d(100, 100, 3, padding=1)

        # add fully connected layer here
        self.fc1 = nn.Linear(100 * 1 * 1, 64)
        self.fc2 = nn.Linear(64, 5)
        # self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 100 * 1 * 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
# class Net(nn.Module):
#     def __init__(self, C=CHANNELS, dropout_p=0.2):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(C, 100, 3, padding=1)
#         self.conv2 = nn.Conv2d(100, 100, 3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv3 = nn.Conv2d(100, 100, 3, padding=1)
#         self.conv3_1 = nn.Conv2d(100, 100, 3, padding=1)
#
#         self.conv3_2 = nn.Conv2d(100, 100, 3, padding=1)
#         self.conv3_3 = nn.Conv2d(100, 100, 3, padding=1)
#         self.conv4 = nn.Conv2d(100, 100, 3, padding=1)
#         self.fc1 = nn.Linear(100 * 1 * 1, 5)
#         # self.dropout = nn.Dropout(dropout_p)
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = F.relu(self.conv3(x))
#         x = F.relu(self.conv3_1(x))
#         x = self.pool(F.relu(self.conv4(x)))
#         x = F.relu(self.conv3_2(x))
#         # x = self.dropout(x)
#         x = F.relu(self.conv3_3(x))
#         x = x.view(-1, 100 * 1 * 1)
#         x = self.fc1(x)
#         return x


def make_onehot_data(inputs, labels):
    """
    Preprocesses data to one hot format to feed into model
    """

    inputs = torch.from_numpy(inputs).float()#.to(torch.long)
    labels = torch.from_numpy(labels).float()

    base = inputs[:, 0:1, :, :] #this contains all elts except next train/targets
    targets = inputs[:, 2:3, :, :]

    mask_layer = (base != 0).float()
    agent_layer = (base == 1).float()
    train_layer = (base == 2).float()
    next_train_layer = inputs[:, 1:2, :, :]
    switch_layer = (base == 3).float()
    object1_layer = (base == 4).float()
    target1_layer = (targets == 1).float()
    object2_layer = (base == 5).float()
    target2_layer = (targets == 2).float()

    onehot_inputs = torch.cat((mask_layer, agent_layer, train_layer, next_train_layer,switch_layer,object1_layer,target1_layer,object2_layer, target2_layer), dim=1)

    if cuda:
        onehot_inputs = onehot_inputs.cuda()
        labels = labels.cuda()

    B = inputs.shape[0]
    onehot_train_inputs = onehot_inputs[:9*B//10]
    train_labels = labels[:9*B//10]

    onehot_test_inputs = onehot_inputs[9*B//10:]
    test_labels = labels[9*B//10:]

    return onehot_train_inputs, onehot_test_inputs, train_labels, test_labels



def train(grids_file="grids_data.npy",actions_file="actions_data.npy",num_epochs=20, C=CHANNELS):
    '''
    C is the number of channels in input array
    C = overall pos mask, agent, empty, obj1, obj2, train, train_next, switch, target1, target2
    '''
    batch_size = 1000

    xs = np.load(grids_file)
    ys = np.load(actions_file)

    onehot_train_xs, onehot_test_xs, train_ys, test_ys = make_onehot_data(xs, ys)


    net = Net(C)
    criterion = nn.MSELoss()#CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001) #original lr 0.001

    if cuda:
        net = net.cuda()

    start = time.time()
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = []

        #trains on the remainder if not gully divisible
        for j in range((len(train_ys)-1)//batch_size+1):
            inputs, labels = onehot_train_xs[batch_size*j:batch_size*(j+1)], train_ys[batch_size*j:batch_size*(j+1)]

            # labels = torch.argmax(labels, dim=1).long()
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss.append(loss.item())
        if epoch%10==0 or (epoch == num_epochs-1):
            test_loss = []
            test_accuracy = []
            with torch.no_grad():
                count = 0
                for j in range((len(test_ys)-1)//batch_size+1):
                    inputs, labels = onehot_test_xs[batch_size*j:batch_size*(j+1)], test_ys[batch_size*j:batch_size*(j+1)]

                    # forward + backward + optimize
                    outputs = net(inputs)
                    label_argmax = torch.argmax(labels, dim=1)
                    output_argmax = torch.argmax(outputs, dim=1)
                    # labels = label_argmax.long()
                    accuracy = torch.mean((label_argmax==output_argmax).float())
                    value_captured = math.exp(labels[output_argmax])/(math.exp(labels[label_argmax]))
                    test_accuracy.append(accuracy.item())

                    loss = criterion(outputs, labels)
                    test_loss.append(loss.item())

            print('Epoch:{}, train loss: {:.3f}, test loss: {:.3f}, test accuracy: {:.3f}'.format(epoch + 1, np.mean(running_loss), np.mean(test_loss), np.mean(test_accuracy)))
            running_loss = 0.0
    print("training took", time.time() - start, "seconds")
    torch.save(net.state_dict(), 'nn_model')
    print("Model saved as nn_model")

def load(C=CHANNELS):
    model = Net(C)
    if not cuda:
        model.load_state_dict(torch.load(NN_FILE ,map_location='cpu'))
    else:
        model.load_state_dict(torch.load(NN_FILE))
    return model

def predict(model, state, C=CHANNELS):
    '''
    model: pytorch model, output of load()
    state: 3x5x5 numpy array corresponding to grid
    returns 1x5(num actions) numpy array of Q-values for each action
    '''
    # _, H, W = state.shape
    # onehot_test_xs = torch.zeros([1, C-1, H, W])
    #
    # #state[1] is current observation
    # test_x = torch.from_numpy(state[0]).unsqueeze(0).unsqueeze(1).to(torch.long)
    # onehot_test_xs.scatter_(1, test_x, torch.ones(onehot_test_xs.shape))
    #
    # next_trains = state[1:2]
    # onehot_test_xs = torch.cat((onehot_test_xs, torch.from_numpy(next_trains).unsqueeze(0).float()), dim=1)

    inputs = torch.from_numpy(state).float()#.to(torch.long)

    # NEWER ATTEMPT TO CLEAN UP INPUT

    base = inputs[0:1, :, :] #this contains all elts except next train/targets
    targets = inputs[2:3, :, :]

    mask_layer = (base != 0).float()
    agent_layer = (base == 1).float()
    train_layer = (base == 2).float()
    next_train_layer = inputs[1:2, :, :]
    switch_layer = (base == 3).float()
    object1_layer = (base == 4).float()
    target1_layer = (targets == 1).float()
    object2_layer = (base == 5).float()
    target2_layer = (targets == 2).float()

    onehot_inputs = torch.cat((mask_layer, agent_layer, train_layer, next_train_layer,switch_layer,object1_layer,target1_layer,object2_layer, target2_layer), dim=0)
    onehot_inputs = torch.unsqueeze(onehot_inputs, dim=0)
    outputs = model(onehot_inputs)

    return outputs.detach().numpy()


if __name__ == "__main__":
    #grids = np.ones((49,2,5,5))
    #actions = np.ones((49,5))

    train(grids_file='grids_1000_switch.npy',actions_file='actions_1000_switch.npy', num_epochs=50)
