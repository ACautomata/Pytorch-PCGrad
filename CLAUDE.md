# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This repository is a compact PyTorch reimplementation of PCGrad from “Gradient Surgery for Multi-Task Learning.” The main reusable component is `PCGrad` in `src/pytorch_pcgrad/pcgrad.py`, which wraps an existing `torch.optim` optimizer and exposes the normal `zero_grad()` / `step()` flow plus an `objectives()` / `__call__()` wrapper whose `.backward()` performs PCGrad gradient surgery on a list of per-task losses. `PCGrad` is fully compatible with the `torch.optim.Optimizer` interface — it transparently forwards `param_groups`, `state`, `defaults`, `state_dict()`, `load_state_dict()`, and `add_param_group()` to the wrapped optimizer. An `objectives()` wrapper returns a `PCGradObjectives` (a `list` subclass) whose `.backward()` routes to the internal `_backward`. `__call__` is aliased to `objectives()` as a shorthand.

The reusable package is intentionally small: `src/pytorch_pcgrad` should expose PCGrad only. Multi-MNIST training code lives under `examples/multi_mnist/` and is not part of the published module surface. It composes a shared representation network (`MultiLeNetR`) and two task-specific output heads (`MultiLeNetO`) from `examples/multi_mnist/net/lenet.py`, creates the Multi-MNIST dataset via `examples/multi_mnist/data/multi_mnist.py`, wraps an Adam optimizer with `PCGrad`, computes left/right digit losses separately, and calls `optimizer(losses).backward()` before `optimizer.step()`.

`examples/multi_mnist/data/multi_mnist.py` adapts the torchvision MNIST dataset flow and generates overlapping two-digit Multi-MNIST tensors under `./dataset/processed` when `download=True`. `depreciate/pcgrad_ori.py` is an older PCGrad implementation kept for reference; prefer editing `src/pytorch_pcgrad/pcgrad.py` for current behavior.

## Commands

Install dependencies:

```bash
conda run -n pcgrad-test python -m pip install -e .
```

Run the Multi-MNIST example training script:

```bash
conda run -n pcgrad-test python examples/multi_mnist/main.py
```

Run the PCGrad regression tests:

```bash
conda run -n pcgrad-test python -m pip install -e .
conda run -n pcgrad-test python -m unittest discover -s tests
```

Run a syntax check across the repository:

```bash
conda run -n pcgrad-test python -m compileall src examples tests depreciate
```

The project uses `pyproject.toml` with hatchling as its build backend. Use the `pcgrad-test` conda environment with targeted test execution and `compileall` for verification.

## Development notes

- The README targets PyTorch 1.6.0 behavior. Be careful with API changes in modern PyTorch/torchvision when editing dataset loading, `torch.load`, or optimizer gradient handling.
- `PCGrad._backward()` expects a list of task losses from the same forward graph and calls backward once per objective with `retain_graph=True`.
- `PCGrad` proxies `param_groups`, `state`, `defaults` (with setters), `state_dict()`, `load_state_dict()`, `add_param_group()`, and uses `__getattr__` to fall through to the wrapped optimizer for any other attribute. `step(closure=None)` and `zero_grad(set_to_none=True)` match PyTorch optimizer semantics.
- `PCGradObjectives` is a `list` subclass with a `.backward()` method that delegates to `optimizer._backward(self)`. Created via `optimizer.objectives(losses)` or `optimizer(losses)`.
- `PCGrad._retrieve_grad()` preserves parameters without gradients by inserting zero tensors and a `has_grad` mask, which matters for multi-head architectures where task-specific heads do not all receive every loss.
- `examples/multi_mnist/main.py` stores experiment hyperparameters as top-level constants (`PATH`, `LR`, `BATCH_SIZE`, `NUM_EPOCHS`, `TASKS`, `DEVICE`) rather than command-line arguments.

## Agent skills

### Issue tracker

Issues and PRDs are tracked in GitHub Issues for this repository. See `docs/agents/issue-tracker.md`.

### Triage labels

Triage uses the default five-label vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, and `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

This is a single-context repository: read root `CONTEXT.md` if present and root `docs/adr/` for architectural decisions. See `docs/agents/domain.md`.
