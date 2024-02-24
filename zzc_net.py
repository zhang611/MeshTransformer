import torch.nn as nn

'''
teddy 5
'''


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(628, 480),
            nn.ReLU(),
            nn.Linear(480, 360),
            nn.ReLU(),
            nn.Linear(360, 240),
            nn.ReLU(),
            nn.Linear(240, 120),
            nn.ReLU(),
            nn.Linear(120, 5),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.model(x)
        return x