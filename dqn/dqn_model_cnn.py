import torch as T
import torch.nn as nn
import torch.optim as optim
import numpy as np


class DeepQNetConv(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, n_actions):
        super(DeepQNetConv, self).__init__()
        # for initialise
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.n_actions = n_actions
        # for gpu
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        # kernel_size depends upon the size of maze
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1),
            # nn.Dropout(p=0.4),
            nn.ReLU()
            # nn.Conv2d(64, 64, kernel_size=2, stride=1),
            # nn.ReLU()
        )
        conv_out_size = self._get_conv_out(self.input_dims)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, self.fc1_dims),
            nn.ReLU(),
            nn.Linear(self.fc1_dims, self.n_actions),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

    def _get_conv_out(self, shape):
      o = self.conv(T.zeros(1, *shape))
      return int(np.prod(o.size()))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        action = self.fc(x)
        return action