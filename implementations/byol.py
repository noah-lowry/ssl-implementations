import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch import nn

from implementations.utils import LARS, DoubleAugmentationDataset, make_linear_to_cosine_scheduler
from implementations.modules import ResNetBase, BYOLProjectionHead

def run_byol_train(dataset, config):
    
    state = make_new_state(dataset, config)
    byol_train_loop(state, config)
    return state

def byol_train_loop(state, config):

    for epoch in (bar := tqdm(range(config["max_epochs"]), desc="Training", position=0, unit="epoch")):
        
        loss = byol_epoch(
            state["loader"],
            state["online"],
            state["target"],
            state["optim"],
            state["tau"],
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

    state = {
        "loader": DataLoader(
            DoubleAugmentationDataset(
                dataset,
                transform=config["transform"] if "transform" in config else None,
                augment_fn=config["augment_fn"],
                device=config["device"],
            ),
            **config["data_kwargs"],
        ),
        "online": nn.Sequential(
            (
                ResNetBase(
                    n_channels=config["base_kwargs"]["n_channels"],
                    model=config["base_class"],
                )
                if isinstance(config["base_class"], str)
                else config["base_class"](**config["base_kwargs"])
            ),
            (
                BYOLProjectionHead(
                    in_features=(config["projection_head_kwargs"]["in_features"]),
                    out_features=(
                        config["projection_head_kwargs"]["out_features"]
                        if "out_features" in config["projection_head_kwargs"]
                        else 256
                    ),
                    hidden_dim=(
                        config["projection_head_kwargs"]["hidden_dim"]
                        if "hidden_dim" in config["projection_head_kwargs"]
                        else 4096
                    ),
                )
                if config["projection_head_class"] == "byol_default"
                else config["projection_head_class"](**config["projection_head_kwargs"])
            ),
            (
                BYOLProjectionHead(
                    in_features=(
                        config["predictor_kwargs"]["in_features"]
                        if "in_features" in config["predictor_kwargs"]
                        else (
                            config["projection_head_kwargs"]["out_features"]
                            if "out_features" in config["projection_head_kwargs"]
                            else 256
                        )
                    ),
                    out_features=(
                        config["predictor_kwargs"]["out_features"]
                        if "out_features" in config["predictor_kwargs"]
                        else 256
                    ),
                    hidden_dim=(
                        config["predictor_kwargs"]["hidden_dim"]
                        if "hidden_dim" in config["predictor_kwargs"]
                        else 4096
                    ),
                )
                if config["predictor_class"] == "byol_default"
                else config["predictor_class"](**config["predictor_kwargs"])
            ),
        ).to(config["device"]),
        "target": nn.Sequential(
            (
                ResNetBase(
                    n_channels=config["base_kwargs"]["n_channels"],
                    model=config["base_class"],
                )
                if isinstance(config["base_class"], str)
                else config["base_class"](**config["base_kwargs"])
            ),
            (
                BYOLProjectionHead(
                    in_features=(config["projection_head_kwargs"]["in_features"]),
                    out_features=(
                        config["projection_head_kwargs"]["out_features"]
                        if "out_features" in config["projection_head_kwargs"]
                        else 256
                    ),
                    hidden_dim=(
                        config["projection_head_kwargs"]["hidden_dim"]
                        if "hidden_dim" in config["projection_head_kwargs"]
                        else 4096
                    ),
                )
                if config["projection_head_class"] == "byol_default"
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
        state["online"].parameters(),
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
    state["tau"] = config["tau_base"]

    return state

def byol_loss(q, z):
    return torch.mean(2 * (1 - torch.bmm(q.unsqueeze(1), z.unsqueeze(2)).squeeze() / (q.norm(dim=1) * z.norm(dim=1))))

def byol_epoch(loader, online, target, optim, tau, verbose_epoch=False):
    """
    ``tau``: temperature tau parameter for the loss function.\n
    Returns the mean contrastive loss.
    """
    online.train()
    target.train()

    losses = []
    
    for batch_idx, batch in enumerate(loader if not verbose_epoch else tqdm(loader, desc="Epoch", leave=False, unit="batch", position=1)):

        x1, x2 = batch
        
        q1 = online(x1)
        z1 = target(x2).detach()
        loss1 = byol_loss(q1, z1)

        q2 = online(x2)
        z2 = target(x1).detach()
        loss2 = byol_loss(q2, z2)
        
        loss = loss1 + loss2
        loss.backward()

        optim.step()
        optim.zero_grad()

        with torch.no_grad():
            params_target = torch.nn.utils.parameters_to_vector(target.parameters())
            params_online = torch.nn.utils.parameters_to_vector(online[:-1].parameters())

            params_target_new = tau*params_target + (1-tau)*params_online

            torch.nn.utils.vector_to_parameters(params_target_new, target.parameters())
        

        losses.append(loss.item())
    
    return np.mean(losses)
