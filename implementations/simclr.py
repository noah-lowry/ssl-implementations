import numpy as np
from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from torch import nn


from implementations.utils import generalized_ntxent, LARS, DoubleAugmentationDataset, make_linear_to_cosine_scheduler
from implementations.modules import ResNetBase, SimCLRProjectionHead


def run_simclr_train(dataset, config):
    
    state = make_new_state(dataset, config)
    simclr_train_loop(state, config)
    return state

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
            DoubleAugmentationDataset(
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
                lambda optim, **kwargs: make_linear_to_cosine_scheduler(
                    optim, config["max_epochs"], **kwargs
                )
            )
            if config["scheduler_class"] == "linear_to_cosine"
            else (
                (
                    lambda optim, **kwargs: torch.optim.lr_scheduler.CosineAnnealingLR(
                        optim, config["max_epochs"], **kwargs
                    )
                )
                if config["scheduler_class"] == "cosine"
                else config["scheduler_class"]
            )
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
    views = torch.cat([torch.stack(item, dim=0) for item in items], dim=0)
    pair_matrix = torch.zeros((len(views), len(views)), dtype=torch.bool)
    rows = torch.arange(0, len(views)-1, 2)
    pair_matrix[rows, rows+1] = pair_matrix[rows+1, rows] = True
    return (views, pair_matrix.to(views.device))
