import torch.nn.functional as F
import torch.nn  as nn


class Net(nn.Module):
    '''
    Represents the structure of pytorch CNN model
    '''
    def __init__(self, N_CLASSES):
        '''
        Describes all layers, contained in the model
        :param N_CLASSES: Number of output classes of the model
        '''
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(3, 8, 2)

        self.conv3 = nn.Conv2d(8, 16, 5)
        self.conv4 = nn.Conv2d(16,32, 5)

        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(32 * 5 * 4, 128)
        self.bnorm1 = nn.BatchNorm1d(128)

        self.fc2 = nn.Linear(128, 64)
        self.bnorm2 = nn.BatchNorm1d(64)

        self.fc3 = nn.Linear(64, N_CLASSES)

    def forward(self, x):
        '''
        Displays the connections between model layers for the forward pass
        :param x: input torch tensor
        :return: model prediction
        '''
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))

        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 32 * 5 * 4)
        x = self.dropout(x)
        x = F.relu(self.bnorm1(self.fc1(x)))
        x = F.relu(self.bnorm2(self.fc2(x)))
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    pass