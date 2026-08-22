# GPU clusters are cheap, but unavailable.. what now?

You're got a great idea on a new model to train or finetune. You've been trying to use a single GPU for as long as possible, to avoid having to deal with all the difficulties of distributed training, but given the data and model sizes it is not possible to continue doing that. So you look around and realize that renting a decent multi-gpu rig is not even that expensive, with a plethora of providers, and marketplaces pooling providers, offering 8 GPUs from the previous generation (A100) for as low as $16 an hour. That's certainly affordable even for learning or play purposes, so you decide to go ahead and rent one.. except that as soon as you actually try to reserve it, you realize there is practically no stock left. Renting a single GPU os any calibre is easy, renting an 8-GPU pod with fast interconnect takes a lot of patience and luck (watcher scripts), and getting a cluster with more than one pods is near impossible. No wonder, given the articles stating that both rent and build capacity is sold out until the end of 2027 at many data centers. 

Any distributed training guide will start by telling how much the whole process is communications-bound, so the question is, what can you do with the left-over capacity that one can scrape? Is a pod with a slower interconnect a waste of money? How much performance can we expect using the most common distributed algorithms with what one can find on the marker right now? As a guiding task, I have picked the nanogtp speedrunning benchmark, which is large enough to benefit from multiple GPUs, small enough to finish in a reasonable time, and widely tested and studied. Part I focuses on data parallelism one a single pod, while Part II will focus on other parallel techniques and multi-pod clusters.

# Setup and baseline

Karpathy started the now-famous nanogpt challenge, which is about training a small GPT2-like model to a a given validation value as fast as possible on a fixed hardware. This led to a community speedrunning competition, in which a series of patches took the original training time from 45 minutes on 8xH100 (90 minutes on 8xA100) down to 1.23 minutes. This is a perfect starting point for our distributed training measurements - the model and data is small enough that I could run tests on my desktop and that cloud measurements don't cost too much, large enough to benefit from multiple GPUs and have a meaningful measurable time, and known enough so that anyone can easily understand what is being measured. In order to keep the model and the code simple and easy to understand, I did not use the fastest entry from the speedrunning leaderboard, as some of the optimizations seemed rather low-level, specific to this model, or I simply had no idea what they were doing. Instead, I chose to start from the original GPT2 baseline and apply some of the most widely understood and tested optimziations from the first entries of the leaderboard, such as using rotary embeddings, using ReLU^2, zero-initializing projections, applying QK-norm and untying embedding and head. However, I decided to skip the Muon optimizer for now. While Muon was integral to the nanogpt-speedrunning competition, it is less widely used and I wanted to keep the most well-known optimizer. This way, my baseline ended up being xx minutes on an A100 and yy minutes on my 3090 desktop card.

Plot 0: Train + val loss curve of a full converged run
Plot 1/a: Bar plot, training runtime to convergence on 1xA100 and 3090 desktop (either actual measured time or calculated from per-step time)
Plot 1/b: Bar plot, estimated training cost for the above
(not sure if a and b should be separate plots or a single plot with two subplots. we will have the same multiple times, with additional bars)

+ each plot shows a single run
+ deterministic data loading is important
+ most measurement are on A100, as the small model did not require larger cards and this was somewhat cheaper

# Ideal setup: DDP over high-speed interconnect

The ideal setup for training on multiple GPUs is an 8-GPU pod with high-speed NVLink interconnect between the cards. The simplest approach of using multiple cards in parallel is data parallelism, in which case each GPU has an identical copy of the model, calculates gradients independently for a different set of the data, then shares the gradients between all GPUs and calculates model updates using the same macrobatch gradient. As we can see from this very short description, this requires a communication step the size of the model before each model update (in each macrobatch), which shows why a fast interconnect is important. Using the ideal setup of 8xA100 GPUs with NVLink interconnect, the model can be trained in only zz minutes.

Same as Plot1 with additional content:
Plot 2/a: Bar plot, training runtime to convergence on 8xA100 in fastest mode (ddp_torch+compile?), plus 1xA100 and 3090 desktop (either actual measured time or calculated from per-step time)
Plot 2/b: Bar plot, estimated training cost for the above

The naive implementation would introduce a long bottleneck after the local computation has finished on each rank, and before the gradients are received from all other ranks, which is the biggest no-no in distributed computing, leading to low GPU utilization and inflated costs. DDP actually has a couple of tricks to reduce this gap, which are explained in detail (with measurements) in Appendix 1. 

# DDP over slower interconnect

8xA100 over NVLink is the ideal setup.. if you can get it. Turns out, despite showcasing per-GPU prices on their homepage, most providers barely have any capacity of proper 8-GPU clusters available.

Image: picture of 'unavailable' texts from RunPod, PrimeIntellect, etc.

If capacity becomes available, it is sometimes using the slower PCIe interconnect between cards, or maybe even the required number of GPUs spread in multiple pods. Should you rent this anyway, or does the slow communication destroy all benefits of having multiple cards and you would be just wasting money? This of course heavily depends on the specific use case you're working with (especially the size of the model and how long processing a single batch takes on the given GPU), so this only gives an indication for this particular problem (very small model). I tried to obtain an 8xA100 PCIe machine and could not, in ways that are themselves part of the story. One provider's PCIe-labelled offer was actually an SXM4 machine with a full NVLink mesh - the socket field turned out to be a catalogue label rather than a fabric guarantee, and every "PCIe" 8xA100 it lists is the same mislabelled SXM4 instance. The other provider does sell the real thing, as a distinct GPU type, at *half* the price of the NVLink box I did rent - and had none of them free. So the transport point I most wanted is the one instance of this article's thesis I could not buy my way out of: it is not expensive, it is unavailable.

What I could rent was **two** A100 80GB PCIe cards, and that turned out to be more informative than expected. The topology is real PCIe - a host-bridge connection, no NVLink anywhere - but the interesting part is what the driver says next: `nvidia-smi topo -p2p r` reports that peer-to-peer is *not supported by the chipset*, and NCCL responds by routing every channel through host shared memory. The GPUs never talk to each other directly at all; every gradient goes out to system RAM and back. Whatever you imagine when you read "PCIe interconnect", this is what renting one actually gives you. The measured all-reduce bandwidth is **2.29 GB/s**, against 151 GB/s on the NVLink box - a factor of 66.

Two things follow. The first is that this is a tax rather than a catastrophe: projected to 8 ranks, the PCIe box needs 2.29 hours to hit the target against 0.94 on NVLink, so it is 2.4x slower - but still 3.1x faster than a single A100. Eight badly connected GPUs beat one well connected one by a wide margin. The second is that this does not make it a bargain. At the rates I was quoted, the NVLink box costs about $21 per converged run and the PCIe box about $25 - the cheaper machine is also the more expensive one, once you pay for the extra hours. The honest answer to "should you rent this anyway" is: yes if it is what is available, but do not go looking for it to save money.

The slow-transport control below is a different thing again, and I keep it because it isolates bandwidth as a variable in a way a rental cannot. The slow-transport control instead disables NCCL's GPU P2P and shared-memory transports and sends the collective through TCP sockets over the same host's loopback interface. That path still crosses the GPU's host link, but it also pays host staging and the TCP/IP software stack; it is therefore a controlled network-transport result, not a substitute for direct GPU-to-GPU PCIe P2P. I then applied netem to that loopback path to simulate lower network bandwidth while keeping the GPUs unchanged.

Plot 3/a: Bar plot, training runtime to convergence on 8xA100 in fastest mode (ddp_torch+compile?) with different interconnects: NVLink, PCIe (measured 2-GPU bandwidth projected to 8), netem 40Gbps, netem 10Gbps, netem 1Gbps
  -> rendered: docs/plots/transport_sensitivity.svg, docs/plots/transport_mfu.svg
Plot 3/b: Bar plot, estimated training cost for the above
maybe it's not really a bar plot that we want here at all? because the cost is influenced by both the rental cost of the given machine type (/hour) and the time it takes, so it could be a two-axis plot as well? not sure.
Plot 4: MFU across the previous options? Showing how much less we can utilize the GPUs under these conditions?

- include conclusions here about how much money one looses/gains by this

# Lowering the communication need with DiLoCo

A reduction in the available communication bandwidth led to a clear underutilization of GPUs. We can't create well-connected GPU pods out of nowhere, so can we do something about the distributed algorithm to reduce the dependence on frequent high-bandwidth communication? When using DDP, the only knob we really have is increasing the batch size, but that becomes undesirable quickly. As we keep increasing the batch size, more and more samples are processed before any update happens to the model. This means we already have a lot of information that tells us what to change in the model (the gradient accumulated so far), yet we keep processing more minibatches with the original model just to be extra certain. It is easy to see how this becomes a waste of computation above a certain size.

An alternative approach is to let the ranks perform updates to their models individually, without having to communicate with other ranks, and only perform some form of synchronization less frquently. There is a multitude of ways of achieving this, the question is how well we can combine these individual changes - do they compound or do they diverge? Distributed Low-communication (DiLoCo) is a method for achieving this, which was shown to work reasonably well. Under DiLoCo, each rank uses the original optimizer to calculate small model updates, which are then shared between ranks less often, and an outer optimizer calculates the global update to the model, which is then shared to each rank. The original paper suggests that this can be used to decrease the communication overhead by 500 times, which is certainly a very appealing prospect. Can we use this to overcome our GPU poorness?

Plot 4: DiLoCo vs original validation loss at identical steps; + ideally DiLoCo full convergence
Plot 5/a: Bar plot, training runtime to convergence on 8xA100 in fastest mode (ddp_torch+compile?), plus 1xA100 and 3090 desktop (either actual measured time or calculated from per-step time); additionally training runtime of DiLoCo to the same steps (which is not full convergence to 3.28, as we don't have that)
Plot 5/b: Bar plot, estimated training cost for the above

As we can see, DiLoCo has worsened the validation result by x% in the end (3.xx instead of 3.28 validation loss). The original training achieved the same validation loss in only yy steps. DiLoCo made training cheaper/more expensive.

We must not forget however that DiLoCo introduced three new parameters to tune: the outer step size H, outer learning rate and moment. Originally I planned to use the values reported in the publication, but those quickly turned out to be unsuitable for this model. We could say that in order to tune this properly, now we need to spend additionaly money on figuring out reasonable values for these parameters, on top of all the other parameters you use for your problem, which means additional cost and uncertainty.

Plot 6/a: Bar plot, training runtime to convergence on 8xA100 in fastest mode (ddp_torch+compile?) with different interconnects: NVLink, PCIe, netem 40Gbps, netem 10Gbps, netem 1Gbps; additionally training runtime of DiLoCo to the same steps (which is not full convergence to 3.28, as we don't have that)
Plot 6/b: Bar plot, estimated training cost for the above
maybe it's not really a bar plot that we want here at all? because the cost is influenced by both the rental cost of the given machine type (/hour) and the time it takes, so it could be a two-axis plot as well? not sure.
Plot 7: MFU across the previous options? Showing how much less we can utilize the GPUs under these conditions?

How much DiLoCo helped under lower communication bandwidth?

+ Maybe a bit more infor about DiLoCo, like the per-rank spread, initial overshoot?

# Conclusion
...

+ include mfu plots somewhere?
+ include theoretical vs measured gpu roofline?

# What's missing 

- additional runs for each parameter with different seeds, to quantify spread
- testing the latest torch, which has improvements in distributed speed

# Appendix
## A1. What makes DDP go brr

- Explain different DDP versions shortly
Runtime comparison of different DDP versions (ddp naive, interleaved, bucketed, torch)
  -> rendered: docs/plots/ddp_mode_comparison.svg (four modes over sockets)
  -> rendered: docs/plots/pcie_modes.svg (compile vs overlap on real PCIe). Worth a paragraph: on a fabric this slow I expected overlap to win, and it does not - the two uncompiled modes land within 0.2% of each other, while compilation alone is worth 1.64x. Overlap can only hide communication that is small enough to hide behind; at 2.29 GB/s it is not.
