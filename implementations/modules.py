from torch import nn
from torchvision.models.resnet import (
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152
)

class ResNetBase(nn.Module):
    
    def __init__(self, n_channels: int, model: str = "resnet50"):
        super().__init__()

        self.n_channels = n_channels
        self.base = resnet18() if model == "resnet18" else \
                    resnet34() if model == "resnet34" else \
                    resnet50() if model == "resnet50" else \
                    resnet101() if model == "resnet101" else \
                    resnet152() if model == "resnet152" else None
        self.base.conv1 = nn.Conv2d(in_channels=self.n_channels, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.out_features = self.base.fc.in_features
        self.base.fc = nn.Identity()
        
    def forward(self, x):
        return self.base(x)


class SimCLRProjectionHead(nn.Module):
    """
    Standard SimCLR projection head.\n
    ### Parameters
    - in_features: determined by output size of base encoder. If using ResNet as base, set to:
        - 512 for resnet[18,34]
        - 2048 for resnet[50,101,152]\n
    - out_features: if specified, projects the output into a different dimension. Defaults to in_features (the paper uses out_features = in_features).
    """
    
    def __init__(self, in_features: int, out_features: int = None):
        
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features or self.in_features

        self.mlp = nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=self.in_features, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=self.in_features, out_features=self.out_features, bias=False)
        )

    def forward(self, x):
        return self.mlp(x)
    
class BYOLProjectionHead(nn.Module):
    """
    Similar to SimCLR projection head but with BN and different defaults.\n
    """

    def __init__(self, in_features: int, out_features: int = 256, hidden_dim: int = 4096):
        
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=self.hidden_dim, bias=False),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_dim, out_features=self.out_features, bias=False)
        )

    def forward(self, x):
        return self.mlp(x)
