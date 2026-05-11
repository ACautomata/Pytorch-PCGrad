# PyTorch-PCGrad

This repository provide code of reimplementation for [Gradient Surgery for Multi-Task Learning](https://arxiv.org/pdf/2001.06782.pdf) in PyTorch 1.6.0. 

## Setup
Install the core package via:
```
pip install -e .
```

`requirements.txt` and the core install only cover the reusable PCGrad package. Install the Multi-MNIST example dependencies via:
```
pip install -e '.[examples]'
```

## Usage

```python
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_pcgrad import PCGrad

# wrap your favorite optimizer
optimizer = PCGrad(optim.Adam(net.parameters()))
losses = [...] # a list of per-task losses
assert len(losses) == num_tasks
optimizer.zero_grad()
optimizer(losses).backward()  # calculate the gradient and apply gradient modification
optimizer.step()  # apply gradient step
```

You can also use the `objectives` wrapper for a `backward`-style call:

```python
optimizer.zero_grad()
losses = optimizer.objectives([loss1, loss2])
losses.backward()
optimizer.step()
```

`PCGrad` is fully compatible with the `torch.optim.Optimizer` interface — `state_dict()`, `load_state_dict()`, `param_groups`, `state`, `defaults`, and `add_param_group()` are all transparently forwarded to the wrapped optimizer.

## Examples
- Mulit-MNIST 
  Please run the example training script via the following command. Part of implementation is leveraged from https://github.com/intel-isl/MultiObjectiveOptimization
  ```
  python examples/multi_mnist/main.py
  ```
  The result is shown below.
  | Method                  | left-digit | right-digit |
  | ----------------------- | ---------: | ----------: |
  | Jointly Training        |      90.30 |       90.01 |
  | **PCGrad (this repo.)** |  **95.00** |   **92.00** |
  | PCGrad (official)       |      96.58 |       95.50 |

- Cifar100-MTL
  coming soon 
## Reference

Please cite as:

```
@article{yu2020gradient,
  title={Gradient surgery for multi-task learning},
  author={Yu, Tianhe and Kumar, Saurabh and Gupta, Abhishek and Levine, Sergey and Hausman, Karol and Finn, Chelsea},
  journal={arXiv preprint arXiv:2001.06782},
  year={2020}
}

@misc{Pytorch-PCGrad,
  author = {Wei-Cheng Tseng},
  title = {WeiChengTseng/Pytorch-PCGrad},
  url = {https://github.com/WeiChengTseng/Pytorch-PCGrad.git},
  year = {2020}
}
```