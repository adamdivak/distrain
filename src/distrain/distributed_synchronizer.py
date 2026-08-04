
import torch
from torch import distributed as dist
from torch import nn


class DistributedSynchronizer:
    def __init__(self, model: nn.Module, mode: str | None, world_size: int):
        self.model = model
        self.world_size = world_size
        self.mode = mode

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

    def finalize_gradients(self):
        # Called after gradients are calculated in each rank. Either does the actual gradient
        # syncing, or waits for the gradients to be synched when using hooks
        if self.mode == "ddp":
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
                # FIXME Assumes identical token count on each rank, we may need to
                # rethink this later?
                param.grad /= self.world_size
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
