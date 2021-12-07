# Explicit_RNN
RNN with customizable layers.

The reason for creating this repository is the following 2 problems:
- the standard PyTorch's [RNN module](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html?highlight=rnn#torch.nn.RNN) doesn't allow you to comfortably change the inner architecture: e.g. add [skip connections](https://arxiv.org/abs/1512.03385) or [layer normalization](https://arxiv.org/abs/1607.06450v1). There is an opportunity for editing [the source code of module](https://pytorch.org/docs/stable/_modules/torch/nn/modules/rnn.html#RNN), but it is too long
- ready-made solutions that I found stopped working when the key parameters were significantly increased

So I write a convenient application for my experiments. Its merits:
- optimized model can be used with any batch size without altering model code itself (e.g. is equal 1 for single sample generation or whatever else for training)
- way the model works is described layer by layer - it's easy to change or add any layer or operation you want (once I added too many skip connections and PyTorch's autograd failed to compute gradient - it might be interesting to calculate this manually)
- project needs minimal modifications to solve other tasks of analysis and sequence generation

To run it, you need the PyTorch library and a simple plain text "input.txt" in the root of the directory. The program periodically saves optimization progress, so you can interrupt the training process at any time. In addition, examples of model generation are stored in a separate folder during the weights tuning.

Thanks to [Andrej Karpathy](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) and [Gabriel Loye](https://github.com/gabrielloye/RNN-walkthrough/blob/master/main.ipynb) for inspiration during my research
