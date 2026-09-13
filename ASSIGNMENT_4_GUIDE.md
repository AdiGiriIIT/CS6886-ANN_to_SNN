# CS6886 Assignment 4 — ANN-to-SNN conversion guide

## Goal

Convert parts (and eventually all) of a pretrained MNIST CNN into a spiking
neural network (SNN), then quantify the trade-off between accuracy, spike
sparsity, and energy. The CNN has three repeated blocks:

`Conv2d -> BatchNorm2d -> ReLU -> AvgPool2d`, followed by `Flatten -> Linear`.

The submission is a reproducible notebook renamed
`<roll-number>_6886A4.ipynb`, with an explanation of the process in the
notebook or a companion PDF. Run **Restart and Run All** immediately before
submission.

## Task map

| Task | Marks | Deliverable | Nature |
|---|---:|---|---|
| 1 | 10 | Working imports after installing the supplied SpikingJelly fork via `python setup.py install` | Setup/verification |
| 2 | 10 | MAC count for every convolution layer | Analytical calculation |
| 3 | 10 | One accuracy-vs-timesteps plot covering every supplied conversion job; select one method for later tasks | Code + experiment |
| 4 | 10 | One accuracy-vs-timesteps plot for the ANN and 1-, 2-, and 3-block conversions | Code + experiment |
| 5 | 10 | Average spike count of every layer in a fully converted SNN | Code + measurement |
| 6 | 15 | Best accuracy whose compute-only energy saving is at least 90% | Analysis/reporting from experiments |
| 7 | 15 | Best compute-only energy saving while accuracy is within 5 percentage points of the ANN | Analysis/reporting from experiments |
| 8 | 5 | Per-layer CNN activation-memory energy for one inference | Analytical calculation |
| 9 | 15 | Best memory-aware energy saving while accuracy is within 5 percentage points of the ANN | Analysis/reporting from experiments |

## Code locations and what to change

Assignment rules permit changes only in the notebook and these two files:

- `spikingjelly/activation_based/ann2snn/examples/cnn_mnist.py`
- `spikingjelly/activation_based/ann2snn/examples/loader.py`

### Notebook: primary work area

Use the notebook for all task-specific experiment drivers, plots, tables, and
explanations.

- **Configuration cell:** change `snn.hyperparameters.T`, `batch_size`,
  `device`, dataset path, and download flags as needed. Sweep `T` to produce
  the requested accuracy/energy trade-offs.
- **Model split cell:** change `split_index` and choose `backbone`/`head` for
  Task 4. The block boundaries are indices `4`, `8`, and `12` in
  `model.network`; `split_index=0` (or `backbone=None`) corresponds to a fully
  converted network.
- **`snn.main(...)` call:** call it once per conversion job and per desired
  split, save the returned `vals`, and plot `vals` against timesteps
  `1..T`.
- **New helper cells:** add MAC, fan-out, activation-size, energy, and plotting
  helpers. This keeps the calculations visible and reproducible.
- **Task 5 driver:** invoke
  `snn.main(custom_val, backbone=None, head=model.network, ...)` and record
  the per-layer counts collected by `custom_val`.

### `cnn_mnist.py`: conversion choices and instrumentation

This is the important library file to edit.

- **`conversion_job(head, train_data_loader)`** currently selects
  `ann2snn.Converter(mode=1.0 / 4, ...)`. Uncomment/use each supplied mode
  (`max`, `99.9%`, `1/2`, `1/4`) or define named conversion-job functions so
  Task 3 can compare all of them.
- **`val(...)`** currently registers a hook only on
  `net.spiking0.if_node`. Copy it into `custom_val(...)` and register one hook
  for *each* converted `IFNode`. Give every entry a unique, stable name (module
  path or layer index); do not use `module.__class__.__name__` alone because
  all IF nodes otherwise collide in the same dictionary key.
- Reset `spike_counts` before each measurement. Register hooks before the
  timestep loop (or otherwise ensure each forward pass is counted exactly
  once), remove them afterward, and calculate spike statistics using
  `train_data_loader` only. Keep accuracy evaluation on `test_data_loader`, as
  the starter comment requires.
- The existing `count_macs_hook` is only a placeholder. It may be replaced by
  a useful hook, but Task 2 is simpler and clearer as an explicit shape-based
  calculation in the notebook.

### `loader.py`: usually leave unchanged

`CustomLoader` runs the frozen CNN backbone and returns its feature maps for a
converted head. It is relevant to partial-conversion experiments (Tasks 3–4),
but no stated task requires modifying it. Edit it only if a real data/device
issue prevents partial-conversion runs; preserve its contract of returning
`(feature_tensor, label)` with the feature tensor detached and on CPU for the
`DataLoader`.

## Calculations to show

### Task 2: CNN MACs

For a convolution with output shape `(C_out, H_out, W_out)`, the MAC count is

`H_out * W_out * C_out * (C_in/groups) * K_h * K_w`.

For the provided valid-padding CNN, the convolution results are:

| Conv layer | Output shape | MACs |
|---|---|---:|
| Block 1 Conv (1 -> 32, 3x3) | `32 x 26 x 26` | 194,688 |
| Block 2 Conv (32 -> 32, 3x3) | `32 x 11 x 11` | 1,115,136 |
| Block 3 Conv (32 -> 32, 3x3) | `32 x 3 x 3` | 82,944 |

The final linear layer uses 320 MACs if it is included in a whole-network
compute total, although Task 2 specifically asks for convolution layers.

### Tasks 5–7: compute-only SNN energy

Use the measured average spikes for each converted layer. Estimate its SNN
accumulates as

`average_spikes_at_layer * fan_out`.

For a standard convolution, a useful fan-out is
`C_out * K_h * K_w` away from borders; state explicitly how border effects are
handled if exact counts are used. Compare the ANN MAC energy with SNN
accumulate energy:

`E_ANN = MACs * 4.6 nJ`

`E_SNN = accumulate_ops * 0.6 nJ`

`energy_saving (%) = 100 * (1 - E_SNN / E_ANN)`.

State whether the final linear layer is included, and use the same convention
for every comparison. For Tasks 6 and 7, search over conversion method,
converted depth, and `T`; report the chosen configuration, accuracy, energy,
and constraint check.

### Task 8: activation-memory energy

The instructions fuse Conv, BatchNorm, ReLU, and AvgPool within each block.
Thus count the *block output* activation map once. With float32 activations:

`writes = ceil(num_output_elements * 4 bytes / 64 bytes)`

`E_memory = writes * 25 nJ`.

The fused-block output maps and resulting costs are:

| Fused block | Output activation | 64-byte writes | Memory energy |
|---|---|---:|---:|
| Block 1 | `32 x 13 x 13` | 338 | 8,450 nJ |
| Block 2 | `32 x 5 x 5` | 50 | 1,250 nJ |
| Block 3 | `32 x 1 x 1` | 2 | 50 nJ |

If accounting for the classifier output as a separate unfused layer, state the
rounding rule and include it consistently. The assignment explicitly says the
same converted-layer memory cost applies **per SNN timestep**; therefore use
`T * E_memory_per_timestep` in Task 9, in addition to compute energy.

## Theory-only vs. run-and-measure work

Strictly analytical tasks are **Task 2** and **Task 8**. They can be computed
by hand or with transparent notebook helpers; neither needs SNN simulation.

Task 1 is installation/verification. Tasks 6, 7, and 9 culminate in reported
numbers, but those numbers must be derived from the simulations and spike
counts from Tasks 3–5. Tasks 3–5 are the core implementation/experimental
work.

## Reproducibility checklist

- Use only PyTorch for ANN logic and SpikingJelly for SNN logic.
- Use the supplied modified fork through `python setup.py install`, not pip.
- Evaluate accuracy only on the test loader; derive hook-based statistics only
  from the train loader.
- Label every plot (method/split, timestep, accuracy) and every energy table
  (assumptions, units, and selected `T`).
- Preserve outputs in execution order and run all notebook cells from a clean
  kernel before submitting.
