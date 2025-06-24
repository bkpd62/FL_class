import torch.nn as nn

class DNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

    def __repr__(self):
        return f"DNN(input_dim={self.net[0].in_features}, hidden_dim={self.net[0].out_features}, output_dim={self.net[-1].out_features})"
