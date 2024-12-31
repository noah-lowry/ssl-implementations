import numpy as np
from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torch import nn
from torchvision.transforms import v2
from torchvision.models.resnet import (
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152
)

from implementations.utils import generalized_ntxent, LARS


def run_simclr_train(dataset, config):
    
    state = make_new_state(dataset, config)
    simclr_train_loop(state, config)
    return state


def make_simclr_augment_fn(image_size=(224, 224), do_color_distort=True, do_blur=True, crop_scale=(0.08, 1), color_strength=1.0, kernel_size_proportion=0.1):
    kernel_size = [int(image_size[0]*kernel_size_proportion), int(image_size[1]*kernel_size_proportion)]
    if kernel_size[0] % 2 == 0:
        kernel_size[0] += 1
    if kernel_size[1] % 2 == 0:
        kernel_size[1] += 1

    return v2.Compose([
        # Crop resize
        v2.RandomResizedCrop(size=image_size, scale=crop_scale),
        v2.RandomHorizontalFlip(p=0.5),

        # Color distort
        v2.Compose([
            v2.RandomApply([v2.ColorJitter(brightness=0.8*color_strength, contrast=0.8*color_strength, saturation=0.8*color_strength, hue=0.2*color_strength)], p=0.8),
            v2.RandomGrayscale(p=0.2),
        ]) if do_color_distort else v2.Identity(),

        # Gaussian blur
        v2.RandomApply([v2.GaussianBlur(kernel_size=kernel_size)], p=0.5) if do_blur else v2.Identity()
    ])


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

        state["monitors"]["epoch"].append(state["monitors"]["epoch"][-1]+1 if len(state["monitors"]["epoch"]) > 0 else 1)
        state["monitors"]["train_loss"].append(loss)

        for callback in config["callbacks"]:
            callback(state, config)
        
        bar.set_postfix({name: history[-1] for name, history in state["monitors"].items()})

def make_new_state(dataset, config):

    # very readable code
    state = {
        "loader": DataLoader(
            SimCLRDataset(
                dataset,
                transform=config["transform"] if "transform" in config else None,
                augment_fn=config["augment_fn"],
                device=config["device"],
            ),
            collate_fn=simclr_dataset_collate_fn,
            **config["data_kwargs"],
        ),
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
                    out_features=(
                        config["projection_head_kwargs"]["out_features"]
                        if "out_features" in config["projection_head_kwargs"]
                        else None
                    ),
                )
                if config["projection_head_class"] == "simclr_default"
                else config["projection_head_class"](**config["projection_head_kwargs"])
            ),
        ).to(config["device"]),
    }
    state["optim"] = (
        torch.optim.AdamW
        if config["optim_class"] == "adamw"
        else (
            torch.optim.SGD
            if config["optim_class"] == "sgd"
            else LARS if config["optim_class"] == "lars" else config["optim_class"]
        )
    )(
        state["model"].parameters(),
        **(
            (
                lambda optim_kwargs, lr_scaling: (
                    optim_kwargs.update(
                        {
                            "lr": optim_kwargs["lr"]
                            * (
                                config["data_kwargs"]["batch_size"] / 256
                                if lr_scaling == "batch_linear"
                                else (
                                    np.sqrt(config["data_kwargs"]["batch_size"])
                                    if lr_scaling == "batch_sqrt"
                                    else None
                                )
                            )
                        }
                    )
                    and None
                )
                or optim_kwargs
            )(
                *(lambda optim_kwargs: (optim_kwargs, optim_kwargs.pop("lr_scaling")))(
                    config["optim_kwargs"].copy()
                )
            )
            if "lr_scaling" in config["optim_kwargs"]
            else config["optim_kwargs"]
        ),
    )
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
    state["monitors"] = {
        "epoch": [],
        "train_loss": [],
        **{name: [] for name in config["monitor_names"]}
    }

    return state

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

def simclr_dataset_collate_fn(items):
    views = torch.cat(items, dim=0)
    pair_matrix = torch.zeros((len(views), len(views)), dtype=torch.bool)
    rows = torch.arange(0, len(views)-1, 2)
    pair_matrix[rows, rows+1] = pair_matrix[rows+1, rows] = True
    return (views, pair_matrix.to(views.device))


class SimCLRDataset(Dataset):
    
    def __init__(self, data, transform, augment_fn, device):
        super().__init__()
        
        self.data = data
        self.transform = transform
        self.augment_fn = augment_fn
        self.device = device
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        selected_data = self.data[idx]
        if self.transform is not None:
            selected_data = self.transform(selected_data)
        return torch.stack([self.augment_fn(selected_data), self.augment_fn(selected_data)], dim=0).to(self.device)


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
