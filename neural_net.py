import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

cuda = True if torch.cuda.is_available() else False
NUM_TARGETS = 2
CHANNELS = 9

class Net(nn.Module):
    def __init__(self, C=CHANNELS):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(C, 100, 3, padding=1)
        self.conv2 = nn.Conv2d(100, 100, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(100, 100, 3, padding=1)
        self.conv4 = nn.Conv2d(100, 100, 3, padding=1)
        self.fc1 = nn.Linear(100 * 1 * 1, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 100 * 1 * 1)
        x = self.fc1(x)
        return x

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
    train_layer = (base == 3).float()
    next_train_layer = inputs[:, 1:2, :, :]
    switch_layer = (base == 4).float()
    object1_layer = (base == 5).float()
    target1_layer = (targets == 1).float()
    object2_layer = (base == 6).float()
    target2_layer = (targets == 2).float()

    onehot_inputs = torch.cat((mask_layer,agent_layer,train_layer, next_train_layer,switch_layer,object1_layer,target1_layer,object2_layer, target2_layer), dim=1)



    # #only keep current trains
    # inputs = inputs[:, 0:1, :, :]
    #
    # B, H, W = inputs.shape[0], inputs.shape[2], inputs.shape[3]
    # # input shape b x c x h x w to net
    # # so need to unsqueeze to make channel dimension of 1, could also make each agent its own channel instead
    # #torch.from_numpy(inputs).unsqueeze(1).to(torch.long)
    # # C - 3 (for targets and next train) + 1 since zero indexed? TODO
    # onehot_inputs = torch.zeros([B, CHANNELS-2, H, W], dtype=torch.float32)
    #
    # #put agent, others, switch, train into their own layer
    # onehot_inputs.scatter_(1, inputs, torch.ones(onehot_inputs.shape))
    #
    # targets = torch.from_numpy(targets).to(torch.long)
    # onehot_targets = torch.zeros([B, NUM_TARGETS+1, H, W], dtype=torch.float32) #+1 for zero index
    # onehot_targets.scatter_(1, targets, torch.ones(onehot_targets.shape))
    # onehot_targets = onehot_targets[:,1:,:,:] #get rid of zero mask
    #
    # #add previous train observations
    # onehot_inputs = torch.cat((onehot_inputs, torch.from_numpy(next_trains).float()), dim=1)
    # #add targets
    # onehot_inputs = torch.cat((onehot_inputs, onehot_targets), dim=1)
    # import pdb; pdb.set_trace()
    # onehot_inputs = torch.cat((onehot_inputs[:,:2,:,:], onehot_inputs[:,3:,:,:]), dim=1)

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
    np.random.shuffle(xs)

    onehot_train_xs, onehot_test_xs, train_ys, test_ys = make_onehot_data(xs, ys)


    net = Net(C)
    criterion = nn.CrossEntropyLoss() #nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    if cuda:
        net = net.cuda()

    start = time.time()
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = []

        #trains on the remainder if not gully divisible
        for j in range((len(train_ys)-1)//batch_size+1):
            inputs, labels = onehot_train_xs[batch_size*j:batch_size*(j+1)], train_ys[batch_size*j:batch_size*(j+1)]
            labels = torch.argmax(labels, dim=1).long()
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

                    accuracy = torch.mean((label_argmax==output_argmax).float())
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
        model.load_state_dict(torch.load('nn_model',map_location='cpu'))
    else:
        model.load_state_dict(torch.load('nn_model'))
    return model

def predict(model, state, C=CHANNELS):
    '''
    model: pytorch model, output of load()
    input_state: 2xHxW(5x5) numpy array
    returns 1x5(num actions) numpy array of Q-values for each action
    '''
    _, H, W = state.shape
    onehot_test_xs = torch.zeros([1, C-1, H, W])

    #state[1] is current observation
    test_x = torch.from_numpy(state[0]).unsqueeze(0).unsqueeze(1).to(torch.long)
    onehot_test_xs.scatter_(1, test_x, torch.ones(onehot_test_xs.shape))

    next_trains = state[1:2]
    onehot_test_xs = torch.cat((onehot_test_xs, torch.from_numpy(next_trains).unsqueeze(0).float()), dim=1)
    outputs = model(onehot_test_xs)

    return outputs.detach().numpy()

if __name__ == "__main__":
    #grids = np.ones((49,2,5,5))
    #actions = np.ones((49,5))

    train(grids_file='shuffled_grids_may28.npy',actions_file='shuffled_actions_may28.npy', num_epochs=50)
