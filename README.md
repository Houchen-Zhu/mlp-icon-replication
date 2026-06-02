# MLP-ICON Local Replication

Note: Some LaTeX formulas may not render correctly in GitHub's web preview. See `math_rendering_note.md` for the recommended viewing method.

## Goal

Implement a simplified MLP-based ICON baseline for the in-context operator learning task:

$$
\left[
  (u^1, Y^1),\ldots,(u^5, Y^5),u^0
\right]
\longmapsto
Y^0 .
$$

Use only the exponential propagator kernel:

$$
G(t)=\exp(-\beta t),
$$

and the linear propagator model:

$$
Y_{t_i}
\approx
\sum_{j=0}^{i}
\exp\!\left[-\beta\left(t_i-t_j\right)\right]
\lambda u_{t_j}\Delta t .
$$

This is a local MacBook-scale replication, not the full Transformer ICON.

---

## Environment Check

Assume VSCode and Python are already installed.

In the VSCode terminal, check:

```bash
python3 --version
pip --version
```

Install packages if needed:

```bash
python3 -m pip install -r requirements.txt
```

Check PyTorch:

```bash
python3 -c "import torch; print(torch.__version__)"
```

Check Apple Silicon MPS:

```bash
python3 -c "import torch; print(torch.backends.mps.is_available())"
```

Use MPS if available, otherwise use CPU.

---

## Recommended Settings

| Item | Value |
|---|---:|
| Kernel | Exponential only |
| Number of regimes, $n_\theta$ | 1,000 |
| Pairs per regime | 10 |
| Time steps, $N$ | 100 |
| Context examples, $M$ | 5 |
| Batch size | 4 |
| Training iterations | 10,000 |
| Learning rate | $10^{-3}$ |
| Model | MLP |

Parameter ranges:

$$
\lambda \sim \mathcal{U}\!\left([0.1,0.5]\right),
\qquad
\beta \sim \mathcal{U}\!\left([0.462,9.011]\right).
$$

---

## Project Structure

```text
mlp_icon_replication/
├── README.md
├── replication_note.md
├── requirements.txt
├── main.py
├── src/
│   ├── config.py
│   ├── data_generation.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── plot_examples.py
└── outputs/
    ├── checkpoints/
    ├── figures/
    └── metrics/
```

`src` means source code. Put all Python modules there.

---

## Data Generation

Implement `src/data_generation.py`.

Generate:

- $n_\theta=1000$ regimes;
- each regime has $\theta_i=(\lambda_i,\beta_i)$;
- for each regime, generate 10 smooth positive selling-rate paths $u$;
- compute the corresponding path $Y=I_{\theta_i}(u)$.

Use:

$$
T=1,
\qquad
N=100,
\qquad
\Delta t=0.01 .
$$

For each $u$, compute:

$$
Y_i
=
\sum_{j=0}^{i}
\exp\!\left[-\beta\left(t_i-t_j\right)\right]
\lambda u_j\Delta t .
$$

The selling-rate path $u$ should be smooth and positive. For the first version, approximate smooth Gaussian-process sampling by generating random normal noise and smoothing it with a moving-average or convolution filter, then add 0.1.

---

## Dataset Sampling

Implement `src/dataset.py`.

For each training task:

1. Select one regime $\theta_i$.
2. From its 10 pairs, randomly choose 1 pair as question/target.
3. From the remaining 9 pairs, randomly choose 5 context pairs.
4. Flatten input:

$$
\left[
u^1,Y^1,
u^2,Y^2,
\ldots,
u^5,Y^5,
u^0
\right].
$$

Input dimension:

$$
5\times(100+100)+100=1100 .
$$

Target:

$$
Y^0 .
$$

Target dimension:

$$
100 .
$$

Return:

```python
x_input: torch.Tensor   # shape [1100]
y_target: torch.Tensor  # shape [100]
```

Optional metadata may include:

```python
lambda_value
beta_value
regime_index
question_index
context_indices
```

---

## Model

Implement `src/model.py`.

Create class:

```python
class MLPICON(nn.Module):
    ...
```

Architecture:

```text
Input: 1100
Linear: 1100 -> 512
GELU
Linear: 512 -> 512
GELU
Linear: 512 -> 256
GELU
Linear: 256 -> 100
```

No output activation.

The model is a simplified MLP-based ICON baseline:

$$
\mathrm{MLP}_{\phi}:\mathbb{R}^{1100}\longrightarrow\mathbb{R}^{100}.
$$

It receives 5 context $(u,Y)$ pairs and one query $u^0$, then predicts $\widehat{Y}^{\,0}$.

---

## Training

Implement `src/train.py`.

Use:

```python
torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
```

Loss:

$$
\mathcal{L}_{\mathrm{MSE}}
=
\frac{1}{N}
\sum_{i=0}^{N-1}
\left(
  \widehat{Y}_i-Y_i
\right)^2 .
$$

Use:

```text
batch_size = 4
num_iterations = 10000
```

Save final checkpoint:

```text
outputs/checkpoints/mlp_icon_final.pt
```

Print training loss every 200 iterations.

Also save occasional checkpoints every 1,000 iterations:

```text
outputs/checkpoints/mlp_icon_step_{iteration}.pt
```

Device selection:

```python
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
```

---

## Evaluation Method 1: Relative L2 Error

Implement `src/evaluate.py`.

Generate fresh unseen test regimes:

```text
n_regimes_test = 200
pairs_per_regime = 10
```

Use the same parameter ranges:

$$
\lambda \sim \mathcal{U}\!\left([0.1,0.5]\right),
\qquad
\beta \sim \mathcal{U}\!\left([0.462,9.011]\right).
$$

Run at least 500 test tasks.

For each test task:

1. Select one unseen test regime.
2. Select 5 context pairs.
3. Select 1 question path.
4. Predict $\widehat{Y}^{\,0}$.
5. Compare against true $Y^0$.

Compute:

$$
\mathrm{RelL2}
\left(
  \widehat{Y},Y
\right)
=
\frac{
  \left\| \widehat{Y}-Y \right\|_2
}{
  \left\| Y \right\|_2
}.
$$

In code:

```python
rel_l2 = torch.norm(y_pred - y_true) / (torch.norm(y_true) + 1e-8)
```

Report:

```text
mean_relative_l2
std_relative_l2
median_relative_l2
min_relative_l2
max_relative_l2
```

Save to:

```text
outputs/metrics/evaluation_metrics.csv
```

---

## Evaluation Method 2: True vs Predicted Plots

Implement `src/plot_examples.py`.

For 5 test tasks, plot:

- true $Y^0(t)$;
- predicted $\widehat{Y}^{\,0}(t)$.

Save to:

```text
outputs/figures/true_vs_pred_0.png
outputs/figures/true_vs_pred_1.png
outputs/figures/true_vs_pred_2.png
outputs/figures/true_vs_pred_3.png
outputs/figures/true_vs_pred_4.png
```

Each plot should include:

- title with $\lambda$, $\beta$, and relative L2 error;
- x-axis: time $t$;
- y-axis: price impact;
- legend: True Y, Predicted Y.

---

## Evaluation Method 3: Context/Question Diagnostic Plot

Create one figure with three panels.

Panel 1:

- 5 context $u$ paths;
- question $u^0$.

Panel 2:

- 5 context $Y$ paths.

Panel 3:

- true $Y^0$;
- predicted $\widehat{Y}^{\,0}$.

Save to:

```text
outputs/figures/context_question_prediction.png
```

This diagnostic plot checks whether the prediction task is visually sensible.

---

## Main Script

Implement `main.py`.

Basic usage:

```bash
python main.py --mode train
python main.py --mode evaluate
python main.py --mode plot
```

This trains the MLP, evaluates it on fresh test regimes, and saves prediction plots under the default `outputs/` directory.

To generate a new batch of synthetic data and save the new run separately, use a different random seed and output directory:

```bash
python main.py --mode train --output_dir output_3 --seed 123
python main.py --mode evaluate --output_dir output_3 --seed 123
python main.py --mode plot --output_dir output_3 --seed 123
```

Here, `--seed 123` changes the randomly generated training/test/plotting data, and `--output_dir output_3` keeps the new results separate from previous runs.

Other CLI parameters that can be changed from the terminal:

```text
--n_regimes_train
--n_regimes_test
--pairs_per_regime
--n_steps
--context_size
--num_iterations
--lr
--seed
--output_dir
--checkpoint_path
```

The batch size is fixed to 4 in this replication.

Default values match the recommended settings.

---

## Success Criteria

For this first local replication:

| Mean Relative L2 Error | Interpretation |
|---:|---|
| $>0.30$ | Not learning well |
| $0.10$-$0.30$ | Learning but rough |
| $0.05$-$0.10$ | Good first replication |
| $<0.05$ | Strong MLP baseline |

Target:

$$
\text{mean\_relative\_l2}<0.10 .
$$

If the error is much larger than 0.30, debug in this order:

1. Check whether $Y$ generation uses only past $u_j$, not future $u_j$.
2. Check whether context pairs and question pair come from the same regime.
3. Check input dimension: should be 1100 for $M=5,N=100$.
4. Check target dimension: should be 100.
5. Reduce learning rate from $10^{-3}$ to $3\times10^{-4}$.
6. Increase training iterations.

---

## Important Conceptual Notes

This MLP model is not the full ICON architecture from the paper. It is a simplified local baseline.

The full ICON uses a Transformer-based architecture and attention masks to enforce causality and handle variable-length time grids.

This first version fixes:

- one kernel family: exponential;
- fixed grid size: $N=100$;
- fixed context size: $M=5$;
- flattened input representation;
- MLP instead of Transformer.

The purpose is to verify the core training logic:

$$
\{(u^m,Y^m)\}_{m=1}^{5}
\;+\;
u^0
\quad\Longrightarrow\quad
Y^0 .
$$

Once this works, the next step is to replace the MLP with a small Transformer model.

---

## Expected Output

After running the full pipeline, the project should produce:

```text
outputs/checkpoints/mlp_icon_final.pt
outputs/metrics/evaluation_metrics.csv
outputs/figures/true_vs_pred_0.png
outputs/figures/true_vs_pred_1.png
outputs/figures/true_vs_pred_2.png
outputs/figures/true_vs_pred_3.png
outputs/figures/true_vs_pred_4.png
outputs/figures/context_question_prediction.png
```

The final console output should include something like:

```text
Evaluation results:
Mean relative L2 error: ...
Std relative L2 error: ...
Median relative L2 error: ...
Min relative L2 error: ...
Max relative L2 error: ...
```

---

## Implementation Order

Implement in this order:

1. `data_generation.py`
2. `dataset.py`
3. `model.py`
4. `train.py`
5. `evaluate.py`
6. `plot_examples.py`
7. `main.py`

Do not implement Transformer yet.

Do not implement OCnet yet.

Do not implement optional normalization yet.

Keep the code simple and readable. Add comments explaining tensor shapes.
