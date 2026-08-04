
import torch
from torch import distributed as dist
from torch import nn


class DistributedSynchronizer:
    def __init__(self, model: nn.Module, mode: str | None, world_size: int,
                 bucket_size: int | None = None):
        self.model = model
        self.world_size = world_size
        self.mode = mode

        if self.mode == "ddp_bucketed":
            if bucket_size is None:
                raise ValueError("ddp_bucketed requires a bucket_size (bytes)")
            self.buckets = []
            self.bucket_size = bucket_size
            self._build_buckets()

        # using torch.no_grad to avoid
        # "an autograd kernel was not registered to the Autograd key(s) but we are trying to backprop through it"
        # warnings. Honestly it's a bit too much PyTorch internals
        # for me to fully understand how this works, but it does
        with torch.no_grad():
            # Distribute model parameters from rank 0 to all other ranks
            # to ensure all start from the same state, regardless of random
            # seeds and different initialization
            for param in self.model.parameters():
                dist.broadcast(param, src=0)

            for buffer in self.model.buffers():
                dist.broadcast(buffer, src=0)

    def _build_buckets(self):
        """Group parameters into all-reduce buckets, in reverse registration order.

        `model.parameters()` delegates to `named_parameters(remove_duplicate=True)`,
        so the tied `wte`/`lm_head` tensor is yielded once and lands in exactly one
        bucket -- the same deduplication the naive path gets, obtained here rather
        than at reduce time.

        A parameter larger than `bucket_size` cannot be split, so it gets a bucket to
        itself. At 124M that is `wte` (154 MB, ~31% of all gradient bytes) sitting
        alone in the *last* bucket, which is worth remembering when mode 3 arrives:
        it is also the last gradient backward produces, so there is no compute left
        to overlap it against.

        Buckets are not keyed by dtype or device, which is fine only while every
        parameter is fp32 on one device -- `_flatten_dense_tensors` concatenates, so
        a mixed bucket would break the moment that stops being true.
        """
        current_bucket = []
        current_size = 0

        # Iterate over the parameters backwards (will be important when attaching hooks)
        for param in list(self.model.parameters())[::-1]:
            size = param.numel() * param.element_size()

            if current_bucket and current_size + size > self.bucket_size:
                self.buckets.append(current_bucket)
                current_bucket = []
                current_size = 0

            current_bucket.append(param)
            current_size += size

        if current_bucket:
            self.buckets.append(current_bucket)

    def finalize_gradients(self):
        # Called after gradients are calculated in each rank. Either does the actual gradient
        # syncing, or waits for the gradients to be synched when using hooks
        if self.mode == "ddp_naive":
            # Iterating over named_parameters also performs deduplication of tied tensors
            # to avoid unnecessary transfer costs
            for _, param in self.model.named_parameters(remove_duplicate=True):
                # We must ensure that each rank calls all_reduce for the same parameters
                # in the same order. In order to avoid issues where one rank has a 
                # gradient for a parameter and the other doesn't, we iterate over all
                # all parameters
                if param.grad is None:
                    param.grad = torch.zeros_like(param)
                dist.all_reduce(param.grad, dist.ReduceOp.SUM)
                # dist.ReduceOp.AVG is not implemented for gloo, so we manually average
                # Note: Assumes identical token count on each rank, currently provided by the data loader
                param.grad /= self.world_size
        elif self.mode == "ddp_bucketed":
            # Bucket membership and order were fixed in the constructor, so every rank
            # reduces the same tensors in the same sequence. Tied parameters were
            # already deduplicated there, by `_build_buckets`
            for bucket in self.buckets:
                # Same rule as the naive path: a missing gradient means the parameter
                # was absent from the autograd graph, so its true contribution to the
                # global mean is zero. Materialise it *on the parameter*, not just in
                # a local list, or the copy-back below has nothing to write into
                for param in bucket:
                    if param.grad is None:
                        param.grad = torch.zeros_like(param)

                grads = [param.grad for param in bucket]
                flat_grads = torch._utils._flatten_dense_tensors(grads)
                dist.all_reduce(flat_grads, dist.ReduceOp.SUM)
                # dist.ReduceOp.AVG is not implemented for gloo, so we manually average.
                # In place on the flat buffer: dividing per parameter instead would
                # allocate a second full set of gradients every step, which is exactly
                # the overhead bucketing exists to remove
                # Note: Assumes identical token count on each rank, currently provided by the data loader
                flat_grads /= self.world_size

                unflat_grads = torch._utils._unflatten_dense_tensors(flat_grads, grads)
                for grad, reduced in zip(grads, unflat_grads):
                    grad.copy_(reduced)
        else:
            raise NotImplementedError()

    def broadcast_scalar(self, value: float, src: int = 0) -> float:
        """Give every rank `src`'s value of a Python scalar.

        Used for the validation loss, which only rank 0 computes. Every rank must
        come back with the *identical* float, because each one independently tests it
        against the target loss -- if src kept its own value and the others took the
        broadcast one, the two could differ in the last bits and ranks could disagree
        about which step first crossed 3.28. So src reads its result back out of the
        tensor too, rather than returning what it passed in.

        The tensor is allocated on the model's device on purpose: NCCL cannot
        broadcast a CPU tensor, and no test in this repo can catch that, since every
        multi-rank test runs on gloo (which accepts both). Taking the device from a
        parameter means it cannot drift from where the model actually lives.
        """
        device = next(self.model.parameters()).device
        tensor = torch.tensor([value], dtype=torch.float32, device=device)
        dist.broadcast(tensor, src=src)
        return tensor.item()

    def wait_for_all_ranks(self) -> None:
        """Block until every rank has finished all of its collectives.

        Called once at the end of training, before any rank can tear its process
        group down. Without it, ranks leave the last collective at slightly different
        times -- and since only rank 0 evaluates, the others leave the final val
        broadcast first, destroy the process group and exit while rank 0 is still
        completing its side of that same broadcast. Rank 0's peer connection then
        drops mid-teardown and gloo aborts the process (SIGABRT, "terminate called
        without an active exception") after training has otherwise succeeded.

        Deliberately at the end of a *successful* run rather than inside
        `cleanup_distributed`: on the error path the surviving ranks would block here
        until the gloo timeout, turning one rank's crash into a hang.
        """
        dist.barrier()
