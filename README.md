# Spiking-Neural-Network

## WORK IN PROGRESS

Building a spiking neural network based on the paper here: https://arxiv.org/abs/1907.13223

Spiking neural networks are similar in structure to standard neural networks, but each node in a spiking network encodes its output temporally as a spike time. For classification problems, this results in an output layer where the node that spikes first indicates the class prediction of the network.

Spiking networks are more biologically plausible than standard neural networks for a couple reasons: 
1. Similar to biological networks in the brain, spiking networks encode information temporally, with relative spike times of the nodes encoding important information.
2. The activation function of each node in the spiking network mimics the dynamics of a real neuron's activation; only a set of temporally close "presynaptic" spikes (inputs to a node) of sufficient strength will cause a node in the spiking network to reach its threshhold and produce a postsynaptic spike of its own.

While temporal encoding in spiking networks brings artificial computation closer to biological networks, there are other differences that need to be addressed. Artificial spiking networks still have a feed forward, "fully" connected architecture where each layer is only connected to the layer directly before and after it. Biological networks, on the other hand, do not have a linear architecture and have neurons connecting to previous layers, forward layers, etc. Additionally, neurons in biological networks are sparsely connected to other neurons in the network.
