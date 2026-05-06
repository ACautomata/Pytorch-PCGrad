# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This repository is a compact PyTorch reimplementation of PCGrad from “Gradient Surgery for Multi-Task Learning.” The main reusable component is `PCGrad` in `pcgrad.py`, which wraps an existing `torch.optim` optimizer and exposes the normal `zero_grad()` / `step()` flow plus `pc_backward(objectives)` for a list of per-task losses.

The only full training script is `main_multi_mnist.py`. It builds a Multi-MNIST experiment by composing three modules: a shared representation network (`MultiLeNetR`) and two task-specific output heads (`MultiLeNetO`) from `net/lenet.py`. The script creates the Multi-MNIST dataset via `data/multi_mnist.py`, wraps an Adam optimizer with `PCGrad`, computes left/right digit losses separately, and calls `optimizer.pc_backward(losses)` before `optimizer.step()`.

`data/multi_mnist.py` adapts the torchvision MNIST dataset flow and generates overlapping two-digit Multi-MNIST tensors under `./dataset/processed` when `download=True`. `depreciate/pcgrad_ori.py` is an older PCGrad implementation kept for reference; prefer editing `pcgrad.py` for current behavior.

## Commands

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run the Multi-MNIST training script:

```bash
python3 main_multi_mnist.py
```

Run the lightweight PCGrad self-check embedded in `pcgrad.py`:

```bash
python3 pcgrad.py
```

Run a syntax check across the repository:

```bash
python3 -m compileall pcgrad.py main_multi_mnist.py data net utils.py depreciate
```

There is no configured test runner, lint command, package metadata, or CI workflow in this repository. Use targeted script execution and `compileall` for verification unless you add a dedicated test setup.

## Development notes

- The README targets PyTorch 1.6.0 behavior. Be careful with API changes in modern PyTorch/torchvision when editing dataset loading, `torch.load`, or optimizer gradient handling.
- `PCGrad.pc_backward()` expects a list of task losses from the same forward graph and calls backward once per objective with `retain_graph=True`.
- `PCGrad._retrieve_grad()` preserves parameters without gradients by inserting zero tensors and a `has_grad` mask, which matters for multi-head architectures where task-specific heads do not all receive every loss.
- `main_multi_mnist.py` stores experiment hyperparameters as top-level constants (`PATH`, `LR`, `BATCH_SIZE`, `NUM_EPOCHS`, `TASKS`, `DEVICE`) rather than command-line arguments.

## Agent skills

### Issue tracker

Issues and PRDs are tracked in GitHub Issues for this repository. See `docs/agents/issue-tracker.md`.

### Triage labels

Triage uses the default five-label vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, and `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

This is a single-context repository: read root `CONTEXT.md` if present and root `docs/adr/` for architectural decisions. See `docs/agents/domain.md`.
