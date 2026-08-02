
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
