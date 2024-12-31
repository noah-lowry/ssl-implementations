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
            "momentum": 0.1875,  # equivalent to 0.9 from the paper because pytorch does momentum weirdly
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
            "momentum": 0.1875,  # equivalent to 0.9 from the paper because pytorch does momentum weirdly
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