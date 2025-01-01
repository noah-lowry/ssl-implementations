import numpy as np

def get_default_simclr_config():
    config = {
        "device": "cuda",
        "max_epochs": 100,
        "verbose_epoch": True,
        
        "data_kwargs": {
            "batch_size": 4096,
            "drop_last": True,
        },
        "transform": lambda x: x[0],
        # "augment_fn": make_simclr_augment_fn(
        #     image_size=(28, 28),
        #     do_color_distort=False,
        #     crop_scale=(0.2, 1)
        # ),

        "base_class": "resnet50",
        "base_kwargs": {
            "n_channels": 1,
        },
        "projection_head_class": "simclr_default",
        "projection_head_kwargs": {
            "in_features": 2048
        },

        "tau": 0.5,
        "optim_class": "lars",
        "optim_kwargs": {
            "lr": 0.3,
            "lr_scaling": "batch_linear",
            "weight_decay": 1e-6,
            "trust_coefficient": 0.001
        },
        "scheduler_class": "linear_to_cosine",
        "scheduler_kwargs": {
            "warmup_iters": 10
        },

        "callbacks": [],
        "monitor_names": []
    }
    return config


def get_mini_simclr_config():
    config = {
        "device": "cuda",
        "max_epochs": 200,
        "verbose_epoch": True,
        
        "data_kwargs": {
            "batch_size": 256,
            "drop_last": True,
        },
        "transform": lambda x: x[0],
        # "augment_fn": make_simclr_augment_fn(
        #     image_size=(28, 28),
        #     do_color_distort=False,
        #     crop_scale=(0.2, 1)
        # ),

        "base_class": "resnet50",
        "base_kwargs": {
            "n_channels": 1,
        },
        "projection_head_class": "simclr_default",
        "projection_head_kwargs": {
            "in_features": 2048
        },

        "tau": 0.2,
        "optim_class": "lars",
        "optim_kwargs": {
            "lr": 0.075,
            "lr_scaling": "batch_sqrt",
            "weight_decay": 1e-6,
            "trust_coefficient": 0.001
        },
        "scheduler_class": "linear_to_cosine",
        "scheduler_kwargs": {
            "warmup_iters": 10
        },

        "callbacks": [],
        "monitor_names": []
    }
    return config

def _byol_tau_cosine_schedule(state, config):
    k = state["monitors"]["epoch"][-1]
    K = config["max_epochs"]
    state["tau"] = 1 - (1-config["tau_base"])*(np.cos(np.pi*k/K)+1)/2

def get_default_byol_config():
    config = {
        "device": "cuda",
        "max_epochs": 1000,
        "verbose_epoch": True,
        
        "data_kwargs": {
            "batch_size": 4096,
            "drop_last": True,
        },
        "transform": lambda x: x[0],

        "base_class": "resnet50",
        "base_kwargs": {
            "n_channels": 1,
        },
        "projection_head_class": "byol_default",
        "projection_head_kwargs": {
            "in_features": 2048,
            "out_features": 256,
            "hidden_dim": 4096
        },
        "predictor_class": "byol_default",
        "predictor_kwargs": {
            "in_features": 256,
            "out_features": 256,
            "hidden_dim": 4096
        },

        "tau_base": 0.996,
        "optim_class": "lars",
        "optim_kwargs": {
            "lr": 0.2,
            "lr_scaling": "batch_linear",
            "weight_decay": 1.5e-6,
            "trust_coefficient": 0.001
        },
        "scheduler_class": "cosine",
        "scheduler_kwargs": {},

        "callbacks": [_byol_tau_cosine_schedule],
        "monitor_names": []
    }
    return config

def get_mini_byol_config():
    config = {
        "device": "cuda",
        "max_epochs": 1000,
        "verbose_epoch": True,
        
        "data_kwargs": {
            "batch_size": 512,
            "drop_last": True,
        },
        "transform": lambda x: x[0],

        "base_class": "resnet50",
        "base_kwargs": {
            "n_channels": 1,
        },
        "projection_head_class": "byol_default",
        "projection_head_kwargs": {
            "in_features": 2048,
            "out_features": 256,
            "hidden_dim": 4096
        },
        "predictor_class": "byol_default",
        "predictor_kwargs": {
            "in_features": 256,
            "out_features": 256,
            "hidden_dim": 4096
        },

        "tau_base": 0.9995,
        "optim_class": "lars",
        "optim_kwargs": {
            "lr": 0.4,
            "lr_scaling": "batch_linear",
            "weight_decay": 1.5e-6,
            "trust_coefficient": 0.001
        },
        "scheduler_class": "cosine",
        "scheduler_kwargs": {},

        "callbacks": [_byol_tau_cosine_schedule],
        "monitor_names": []
    }
    return config

