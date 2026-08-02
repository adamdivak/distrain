
from distrain.train import TrainConfig


def tiny_train_config(tiny_data, **overrides):
    cfg = {
        "train_glob": str(tiny_data / "t_train_*.bin"),
        "val_glob": str(tiny_data / "t_val_*.bin"),
        "seq_len": 32,
        "global_batch_seqs": 8,
        "grad_accum_steps": 1,
        "n_layer": 2,
        "n_head": 2,
        "n_embd": 32,
        "vocab_size": 64,
        "warmup_steps": 2,
        "max_steps": 30,
        "learning_rate": 3e-3,
        "min_lr": 3e-4,
        "val_every": 15,
        "val_tokens": 3200,
        "device": "cpu",
        "trackio": False,
        "log_every": 100,
    }
    cfg.update(overrides)
    return TrainConfig(**cfg)
