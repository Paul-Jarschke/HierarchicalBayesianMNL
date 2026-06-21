# Hierarchical Bayesian Multinomial Logit — Mixture Model Comparison

A simulation study comparing two Bayesian HBMNL mixture model implementations:

- **bayesm** (R) — Gibbs sampler with conjugate updates (Rossi, 2006)
- **Liesel / NuPyro** (Python) — NUTS sampler via gradient-based MCMC

The study uses Rossi et al. (2006) as the primary methodological reference.

---

## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) — Python package and project manager
- Python 3.13 (managed automatically by uv via `.python-version`)

---

## Setup

Clone the repository and install all dependencies:

```bash
git clone <repo-url>
cd HIERARCHICALBAYESIANMNL
uv sync
```

`uv sync` reads `pyproject.toml`, creates a virtual environment under `.venv/`, and installs all pinned dependencies from `uv.lock`. No manual `pip install` or `venv` activation needed.

---

## Project Structure

```
HIERARCHICALBAYESIANMNL/
│
├── generate_data.py                    # CLI entry point — generates all datasets
│
├── data/
│   ├── simulated/
│   │   └── mixture/                   # generated JSON datasets land here
│   ├── margarine/                      # real Rossi (2006) margarine panel data
│   └── camera/                         # real camera choice data
│
├── hbmnl_mixture_experiments/          # one subfolder per K scenario
│   ├── __init__.py
│   ├── experiment_configs.py           # single source of truth for all scenarios
│   ├── 1_comp/
│   │   ├── HMC/
│   │   └── NUTS/
│   ├── 2_comp/
│   ├── 3_comp/
│   └── 5_comp/
│
└── src/
    ├── __init__.py
    ├── dgp.py                          # data-generating process
    ├── mixturemodel.py                 # model specification
    └── inference/
        ├── __init__.py
        ├── hmc.py
        ├── nuts.py
        └── iwls.py
```

---

## Generating Simulation Data

All simulation scenarios are defined in `hbmnl_mixture_experiments/experiment_configs.py`. The `generate_data.py` script at the project root reads those configs and writes datasets to `data/simulated/mixture/`.

### List available scenarios

```bash
uv run python generate_data.py --list
```

Output:

```
Available scenarios:
  1comp                     K=1, n_units=300, n_obs=30
  2comp_equal               K=2, n_units=300, n_obs=30
  3comp_equal               K=3, n_units=300, n_obs=30
  5comp_equal               K=5, n_units=300, n_obs=30
```

### Generate all scenarios

```bash
uv run python generate_data.py
```

This writes `1comp.json`, `2comp_equal.json`, `3comp_equal.json`, and `5comp_equal.json` into `data/simulated/mixture/`.

### Generate a single scenario

```bash
uv run python generate_data.py --scenario 2comp_equal
```

---

## Simulation Design

### Scenarios

| Scenario      | K   | n_units | n_obs | pvec                      |
| ------------- | --- | ------- | ----- | ------------------------- |
| `1comp`       | 1   | 300     | 30    | [1.0]                     |
| `2comp_equal` | 2   | 300     | 30    | [0.5, 0.5]                |
| `3comp_equal` | 3   | 300     | 30    | [⅓, ⅓, ⅓]                 |
| `5comp_equal` | 5   | 300     | 30    | [0.2, 0.2, 0.2, 0.2, 0.2] |

The K=1 scenario degenerates to a standard HMNL and serves as a sanity check that both samplers agree on the baseline. Equal mixture weights are used throughout to maximise label-switching pressure — the hardest setting for both samplers.

The sample size (300 DMUs × 30 observations) mirrors Rossi's (2006) margarine example, making results directly interpretable against that benchmark.

### DGP Specification

The data-generating process follows Rossi (2006) §5.5:

```
θᵢ  = Δ'zᵢ + uᵢ
uᵢ  ~ N(μ_indᵢ,  Σ_indᵢ)
indᵢ ~ Multinomial_K(pvec)

μₖ    ~ N(0, I / A_μ),   A_μ = 1/16
Σₖ    ~ IW(ν, V),         ν = nvar+3,  V = ν·I
```

Key DGP choices and their Rossi justification:

- **Z is column-wise centred** — so that the mean of θ at average z is determined entirely by the mixture component means (§5.5)
- **Continuous X is standardised globally** — so that the prior on μₖ is interpretable on a common scale (§5.5)
- **A_μ = 1/16** — Rossi's recommended precision for standardised X, admitting component means within ±8 (2 SD)
- **Σₖ drawn from IW** — full positive-definite matrices, not diagonal, to exercise off-diagonal posterior recovery

---

## Adding a New Scenario

Open `hbmnl_mixture_experiments/experiment_configs.py` and add an entry to `SCENARIOS`:

```python
"2comp_unequal": {
    **BASE,
    "n_components": 2,
    "custom_pvec":  [0.75, 0.25],
},
```

Then run:

```bash
uv run python generate_data.py --scenario 2comp_unequal
```

---

## References

- Rossi, P. E., Allenby, G. M., & McCulloch, R. (2006). _Bayesian Statistics and Marketing_. Wiley.
- Morris, T. P., White, I. R., & Crowther, M. J. (2019). Using simulation studies to evaluate statistical methods. _Statistics in Medicine_, 38(11), 2074–2102.
