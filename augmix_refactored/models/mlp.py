import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, width, depth, in_features, out_features):
        super().__init__()
        self.mlp = []
        self.width = width
        self.depth = depth
        self.in_features = in_features
        self.out_features = out_features
        self.mlp.append(nn.Linear(in_features, width, bias=True))
        for i in range(depth):
            self.mlp.append(nn.Linear(width, width, bias=True))
        self.mlp.append(nn.Linear(width, out_features, bias=True))
        self.mlp = nn.Sequential(*self.mlp)

    def forward(self, x):        
        return self.mlp(x)