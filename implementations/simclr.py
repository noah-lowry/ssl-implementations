import numpy as np
from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from torch import nn
from torchvision.models.resnet import (
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152
)

from utils import generalized_ntxent


def run_simclr_train(dataset, config):

    # very readable code
    state = {
        "loader": DataLoader(dataset, **config["data_kwargs"]),
        "model": nn.Sequential(
            (
                ResNetBase(
                    n_channels=config["base_kwargs"]["n_channels"],
                    model=config["base_class"],
                )
                if isinstance(config["base_class"], str)
                else config["base_class"](**config["base_kwargs"])
            ),
            (
                SimCLRProjectionHead(
                    in_features=(config["projection_head_kwargs"]["in_features"]),
                    out_features=(config["projection_head_kwargs"]["out_features"]),
                )
                if config["projection_head_class"] == "simclr_default"
                else config["projection_head_class"](**config["projection_head_kwargs"])
            ),
        ),
    }
    state["optim"] = (
        torch.optim.AdamW
        if config["optim_class"] == "adamw"
        else (
            torch.optim.SGD if config["optim_class"] == "sgd" else config["optim_class"]
        )
    )(state["model"].parameters(), **config["optim_kwargs"])
    state["scheduler"] = (
        (
            (
                lambda optim, warmup_iters=10, eta_min=0, last_epoch=-1: SequentialLR(
                    optim,
                    schedulers=[
                        LinearLR(
                            optim,
                            start_factor=1e-2,
                            end_factor=1,
                            total_iters=warmup_iters,
                            last_epoch=last_epoch,
                        ),
                        CosineAnnealingLR(
                            optim,
                            T_max=config["max_epochs"] - warmup_iters,
                            eta_min=eta_min,
                            last_epoch=last_epoch,
                        ),
                    ],
                    milestones=[warmup_iters],
                    last_epoch=last_epoch,
                )
            )
            if config["scheduler_class"] == "linear_to_cosine"
            else config["scheduler_class"]
        )(state["optim"], **config["scheduler_kwargs"])
        if "scheduler_class" in config
        else None
    )


def simclr_train_loop(state, config):

    for epoch in (bar := tqdm(range(config["max_epochs"]), desc="Training", position=0, unit="epoch")):
        
        loss = simclr_epoch(
            state["loader"],
            state["model"],
            state["optim"],
            config["tau"],
            verbose_epoch=config["verbose_epoch"]
        )
        if "scheduler" in state and state["scheduler"] is not None:
            state["scheduler"].step()

        state["monitors"]["epoch"].append(state["monitors"]["epoch"]+1 if len(state["monitors"]["epoch"]) > 0 else 1)
        state["monitors"]["train_loss"].append(loss)

        for callback in config["callbacks"]:
            callback(state, config)
        
        bar.set_postfix({name: history[-1] for name, history in state["monitors"].items()})


def simclr_epoch(loader, model, optim, tau, verbose_epoch=False):
    """
    ``tau``: temperature tau parameter for the loss function.\n
    Returns the mean contrastive loss.
    """
    model.train()

    losses = []
    
    for batch_idx, batch in enumerate(loader if not verbose_epoch else tqdm(loader, desc="Epoch", leave=False, unit="batch", position=1)):

        inputs, pair_matrix = batch
        
        embeddings = model(inputs)
        
        loss = generalized_ntxent(embeddings, pair_matrix, tau=tau)
        loss.backward()
        
        optim.step()
        optim.zero_grad()

        losses.append(loss.item())
    
    return np.mean(losses)


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
