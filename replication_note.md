# Replication Note: MLP-Based ICON Baseline

## 1. Objective

This note summarizes a small-scale replication of the ICON training step from the paper *Solving Optimal Execution Problems via In-Context Operator Networks*.

The goal is to verify the core in-context operator learning pipeline:

$$
\left[
  (u^1,Y^1),\ldots,(u^5,Y^5),u^0
\right]
\longmapsto
Y^0 .
$$

Here, $\{(u^m,Y^m)\}_{m=1}^{5}$ are context examples from the same price-impact environment, $u^0$ is a query selling-rate trajectory, and $Y^0$ is the corresponding price-impact trajectory.

This replication focuses only on the ICON-style price-impact operator learning step. It does not yet reproduce the full Transformer ICON architecture or the downstream OCnet optimal control solver.

## 2. Simplification

To make the experiment feasible on a local MacBook, I implemented a simplified MLP-based baseline instead of the full Transformer-based ICON model.

The model takes a flattened input consisting of five context $(u,Y)$ pairs and one query $u^0$, and predicts $\widehat{Y}^{\,0}$.

For $N=100$ time steps and $M=5$ context examples, the input dimension is:

$$
M(2N)+N
=
5\times(100+100)+100
=
1100 ,
$$

and the output dimension is:

$$
N=100 .
$$

## 3. Synthetic Data Generation

I used the exponential linear propagator model:

$$
G(t)=\exp(-\beta t),
$$

with price impact computed by the causal discrete convolution:

$$
Y_{t_i}
\approx
\sum_{j=0}^{i}
\exp\!\left[-\beta\left(t_i-t_j\right)\right]
\lambda u_{t_j}\Delta t .
$$

The experimental setup is:

- $T=1$
- $N=100$
- $\Delta t=0.01$
- $\lambda\sim\mathcal{U}\!\left([0.1,0.5]\right)$
- $\beta\sim\mathcal{U}\!\left([0.462,9.011]\right)$
- 1,000 training regimes
- 10 selling-rate trajectories per regime
- 5 context examples per task
- batch size 4
- 10,000 training iterations

The selling-rate trajectories $u$ are generated as smooth positive paths.

## 4. Model

The MLP baseline has the following architecture:

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

Equivalently, the model is a fixed-grid operator surrogate:

$$
\operatorname{MLP}_{\phi}:\mathbb{R}^{1100}\longrightarrow\mathbb{R}^{100}.
$$

The model is trained using mean squared error between the predicted price-impact path $\widehat{Y}^{\,0}$ and the true path $Y^0$:

$$
\mathcal{L}_{\mathrm{MSE}}
=
\frac{1}{N}
\sum_{i=0}^{N-1}
\left(
  \widehat{Y}^{\,0}_{t_i}
  -
  Y^0_{t_i}
\right)^2 .
$$

## 5. Evaluation

The model is evaluated on unseen test regimes sampled from the same parameter ranges.

For each test task, I generate a new regime $\theta=(\lambda,\beta)$, construct five context examples, provide one query trajectory $u^0$, and compare the predicted price impact $\widehat{Y}^{\,0}$ with the true path $Y^0$.

The main metric is relative $L^2$ error:

$$
\operatorname{RelL2}
\left(
  \widehat{Y},Y
\right)
=
\frac{
  \left\lVert \widehat{Y}-Y \right\rVert_2
}{
  \left\lVert Y \right\rVert_2
}.
$$

In my local runs, the model achieved approximately:

```text
Mean relative L2 error: 0.06
Maximum relative L2 error: 0.28
```

This suggests that the simplified MLP baseline successfully learns the core in-context operator learning task in the reduced exponential-kernel setting.

## 6. Current Limitations

This is only a first-step replication. The current implementation does not include:

- the full Transformer-based ICON architecture;
- causal attention masking;
- variable or irregular time grids;
- power-law propagator kernels;
- the downstream OCnet policy optimization step;
- real market data or $(u,P)$-based noisy execution-price examples.

## 7. Next Steps

The next steps are:

1. Replace the MLP baseline with a small Transformer-based model.
2. Add non-singular and singular power-law propagator kernels.
3. Compare in-distribution and out-of-distribution prediction errors.
4. Use the trained surrogate operator inside the OCnet optimization step.
5. Explore how market-state variables, such as order-book imbalance and aggregate order flow, could be incorporated as additional conditioning information.
