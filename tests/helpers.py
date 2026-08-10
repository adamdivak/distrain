
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
        # flat LR: lr_at asserts warmup + warmdown < max_steps, and tests override
        # max_steps down to 2. The trapezoid's shape is covered by TestLRSchedule;
        # end-to-end tests here are about the loop, not the schedule
        "warmup_steps": 0,
        "warmdown_steps": 0,
        "max_steps": 30,
        "learning_rate": 3e-3,
        "val_every": 15,
        "val_tokens": 3200,
        # set explicitly rather than inherited: the eval batch must be identical across
        # every config being compared, so a test comparing two configs should not let it
        # come from a default that a future edit could couple to the training batch again
        "eval_batch_seqs": 8,
        "device": "cpu",
        "trackio": False,
        "log_every": 100,
        "ddp_bucket_size": 25000, # bytes
    }
    cfg.update(overrides)
    return TrainConfig(**cfg)
