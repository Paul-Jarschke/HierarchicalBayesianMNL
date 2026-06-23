# Hierarchical Bayesian Multinomial Logit — Mixture Model Comparison

A simulation study comparing two Bayesian HBMNL mixture-of-normals implementations:

- **bayesm** (R) — Gibbs sampler with a random-walk Metropolis step for the choice
  coefficients (`rhierMnlRwMixture`, Rossi 2006)
- **Liesel / Goose** (Python) — gradient-based MCMC: NUTS, fixed-step HMC, and a
  mixed IWLS sampler

The two implementations are run on identical datasets so that differences in
posterior recovery and mixing can be attributed to the samplers rather than the
data. Rossi, Allenby & McCulloch (2006) is the primary methodological reference;
Gelman et al. (BDA3) and Morris et al. (2019, ADEMP) inform the diagnostics and
the simulation-study design.

A central feature of the study is that the number of components the **model**
fits (`K_MODEL`) is decoupled from the number of components in the **data**
(`K_TRUE`). This lets the same machinery cover the correctly-specified case
(`K_MODEL = K_TRUE`) and the overspecified case (`K_MODEL > K_TRUE`), where
surplus components must collapse.

---

## Prerequisites

**Python side**

- [uv](https://docs.astral.sh/uv/getting-started/installation/) — Python package and project manager
- Python 3.13 (managed automatically by uv via `.python-version`)

**R side (bayesm replication)**

- R ≥ 4.5
- [renv](https://rstudio.github.io/renv/) — per-project R library, the R analogue of `uv.lock`

---

## Setup

Clone the repository:

```bash
git clone <repo-url>
cd HierarchicalBayesianMNL
```

**Python environment:**

```bash
uv sync
```

`uv sync` reads `pyproject.toml`, creates `.venv/`, and installs all pinned
dependencies from `uv.lock`. No manual `pip install` or venv activation needed —
prefix commands with `uv run`.

**R environment (only needed for the bayesm comparison):**

```r
# from an R session opened at the repo root
renv::restore()
```

`renv::restore()` reads `renv.lock` and installs the exact recorded versions of
`bayesm`, `jsonlite`, `this.path`, and their dependencies into the project-private
library under `renv/library/`, isolated from your global R packages.

---

## Project Structure

```
HierarchicalBayesianMNL/
│
├── generate_data.py                    # CLI — generates all simulation datasets
├── run_single_experiment.py            # runs ONE model fit, saves all output
├── run_all_experiments.py              # batch orchestrator (overnight runs)
│
├── pyproject.toml / uv.lock            # Python dependencies (uv)
├── renv.lock / .Rprofile / renv/       # R dependencies (renv)
│
├── data/
│   ├── simulated/
│   │   └── mixture/                    # generated JSON datasets land here
│   ├── margarine/                      # real Rossi (2006) margarine panel data
│   └── camera/                         # real camera choice data
│
├── batch_logs/                         # master log + manifest.csv per batch run
│
├── hbmnl_mixture_experiments/
│   ├── __init__.py
│   ├── experiment_configs.py           # single source of truth for all scenarios
│   ├── 1_chain/                        # single-chain runs
│   │   ├── 1_comp/  {HMC, NUTS}/results/
│   │   ├── 2_comp/  {HMC, NUTS}/results/
│   │   ├── 3_comp/  {HMC, NUTS}/results/
│   │   └── 5_comp/  {HMC, NUTS}/results/
│   └── 4_chains/                       # four-chain runs (same layout)
│       ├── 1_comp/ … 5_comp/
│
└── src/
    ├── __init__.py
    ├── dgp.py                          # data-generating process
    ├── mixturemodel.py                 # Liesel model specification
    ├── analysis.py                     # diagnostics, recovery plots, export
    └── inference/
        ├── __init__.py
        ├── nuts.py                     # adaptive NUTS runner
        ├── hmc.py                      # fixed-step HMC runner
        └── iwls.py                     # mixed IWLS + NUTS runner (experimental)
```

The interactive notebooks (`liesel_model.ipynb`) and the bayesm `.R` script live
inside the relevant `hbmnl_mixture_experiments/.../{NUTS,HMC}/` folders, alongside
the `results/` directory that batch output is written to.

---

## Generating Simulation Data

All scenarios are defined in `hbmnl_mixture_experiments/experiment_configs.py`.
`generate_data.py` reads those configs and writes datasets to
`data/simulated/mixture/`.

```bash
uv run python generate_data.py --list           # list available scenarios
uv run python generate_data.py                   # generate all scenarios
uv run python generate_data.py --scenario 2comp_equal   # generate one
```

This writes `1comp.json`, `2comp_equal.json`, `3comp_equal.json`, and
`5comp_equal.json`.

---

## Simulation Design

### Scenarios

| Scenario      | K   | n_units | n_obs | pvec                      |
| ------------- | --- | ------- | ----- | ------------------------- |
| `1comp`       | 1   | 300     | 30    | [1.0]                     |
| `2comp_equal` | 2   | 300     | 30    | [0.5, 0.5]                |
| `3comp_equal` | 3   | 300     | 30    | [⅓, ⅓, ⅓]                 |
| `5comp_equal` | 5   | 300     | 30    | [0.2, 0.2, 0.2, 0.2, 0.2] |

The K=1 scenario degenerates to a standard HMNL and serves as a sanity check that
both samplers agree on the baseline. Equal mixture weights are used throughout to
maximise label-switching pressure — the hardest setting for both samplers. The
sample size (300 DMUs × 30 observations) mirrors Rossi's (2006) margarine example.

### DGP Specification

The data-generating process follows Rossi (2006) §5.5:

```
θᵢ   = Δ'zᵢ + uᵢ
uᵢ   ~ N(μ_indᵢ,  Σ_indᵢ)
indᵢ ~ Multinomial_K(pvec)

μₖ   ~ N(0, I / A_μ),            A_μ = 1/16
Σₖ   = diag(σ²₁, …, σ²ₚ),        σ²ⱼ ~ Uniform(0.5, 2.0)
```

Key DGP choices:

- **Z is column-wise centred** — the mean of θ at average z is determined entirely
  by the mixture component means (§5.5).
- **Continuous X is standardised globally** — so the prior on μₖ is interpretable
  on a common scale, consistent with the model the samplers fit (§5.5).
- **A_μ = 1/16** — Rossi's recommended precision for standardised X, admitting
  component means within roughly ±8 (2 SD).
- **Σₖ is diagonal**, with variances drawn from Uniform(0.5, 2.0). The DGP keeps the
  true component covariances diagonal; the _model_ (below) still places a full
  Wishart prior on each Σₖ⁻¹, so off-diagonal posterior recovery is still exercised
  even though the truth is diagonal.

Each generated JSON records the DGP hyperparameters it used (e.g. `DGP_A_MU`)
alongside the data and the ground-truth parameters (`TRUE_MU_K`, `TRUE_SIGMA_K`,
`TRUE_PVEC`, `TRUE_BETA`, `TRUE_DELTA`, `TRUE_INDICATORS`).

---

## Model & Samplers

### Model prior (Liesel and bayesm)

Both implementations fit the same hierarchical prior, matching
`bayesm::rhierMnlRwMixture`:

```
βᵢ = Z[i] Δ + uᵢ,     uᵢ ~ N(μₖ, Σₖ),   k ~ Categorical(pvec)

Σₖ⁻¹ ~ Wishart(ν, V⁻¹),   ν = n_params + 3,   V = ν·I
μₖ | Σₖ ~ N(0, Σₖ / a_μ)
Δ        ~ N(0, (1/A_Δ) · I)
pvec     ~ Dirichlet(dirichlet_a)
```

Default hyperparameters: `a_μ = 0.01`, `A_Δ = 0.01`, `dirichlet_a = 1.0`. Note the
model places a _full-covariance_ Wishart prior on Σₖ⁻¹ regardless of the diagonal
DGP — the model is not told how the data were generated.

### K_MODEL vs K_TRUE

The number of components the model fits is supplied explicitly and is independent
of the data:

- `K_TRUE` is read from the dataset (a property of the data).
- `K_MODEL` is a modelling decision passed to `build_mixture_hbmnl_model(..., K=K_MODEL)`.

Two strategies are supported by the batch runner:

- **`fixed5`** — fit `K_MODEL = 5` on every scenario (the overspecified study; surplus
  components are expected to collapse toward zero weight).
- **`known`** — fit `K_MODEL = K_TRUE` (the correctly-specified baseline).

When `K_MODEL > K_TRUE`, a smaller `dirichlet_a` (e.g. `0.5`) places more prior mass
near the simplex corners and encourages spurious components to shrink.

### Samplers

| Sampler | Module                  | Strategy                                                                                            |
| ------- | ----------------------- | --------------------------------------------------------------------------------------------------- |
| NUTS    | `src/inference/nuts.py` | Adaptive trajectory length; one NUTS kernel per block.                                              |
| HMC     | `src/inference/hmc.py`  | Fixed-length leapfrog (default 10 integration steps) per block.                                     |
| IWLS    | `src/inference/iwls.py` | Mixed: IWLS on coefficient blocks (μₖ, Δ, βᵢ), NUTS on the simplex / Cholesky blocks. Experimental. |

All runners sample five blocks separately — `pvec_latent`,
`sigma_inv_chol_k_latent`, `mu_k`, `Delta` (if demographics present), `beta_i` — and
take an explicit `K` for correct logging.

---

## Running Experiments

Single fits and overnight batches are run as plain Python scripts (not notebooks),
so they can run unattended.

### One experiment

```bash
uv run python run_single_experiment.py \
    --scenario 5comp_equal --k-model 5 --sampler nuts \
    --chains 1 --warmup 2000 --posterior 10000 --seed 42 \
    --outdir hbmnl_mixture_experiments/1_chain/5_comp/NUTS/results/run
```

Writes into `--outdir`:

| File                | Contents                                                                  |
| ------------------- | ------------------------------------------------------------------------- | ---------------------------------- |
| `mcmc_results.pkl`  | Full Goose results object (warmup, tuning, error records, draws).         |
| `posterior_raw.pkl` | Posterior draws for all parameters, as numpy arrays (portable).           |
| `export.pkl`        | μ / Σ / std / pvec draws for marginal-density comparison.                 |
| `sampling.log`      | Clean Goose engine log (epochs + per-kernel error counts).                |
| `summary.txt`       | Human-readable headline: dims, config, timing, per-kernel errors (named). |
| `meta.json`         | Structured config + dimensions + timing + parsed `sampling_errors`.       |
| `status.json`       | `{"status": "success"                                                     | "failed", ...}` — used for resume. |

> Goose's warmup schedule has a minimum length; very small `--warmup` values
> (e.g. 50) raise `warmup_duration too short`. Use `--warmup 200` or more for quick
> smoke tests; production runs use 2000.

### The full batch

`run_all_experiments.py` defines the experiment grid
(`{1, 4} chains × {1,2,3,5} components × {nuts, hmc}`) and runs each fit as a
**separate subprocess**, so JAX memory is released between fits and a hard crash in
one fit cannot kill the batch.

```bash
uv run python run_all_experiments.py --dry-run        # print the plan only
uv run python run_all_experiments.py                  # fixed5 (K_MODEL=5 everywhere)
uv run python run_all_experiments.py --strategy known # fit K_MODEL = K_TRUE
uv run python run_all_experiments.py --force          # re-run completed experiments
```

Behaviour:

- **Resumable** — experiments whose `status.json` reports success are skipped, so a
  re-run after interruption continues where it stopped.
- **Robust** — each subprocess has a wall-clock timeout (`TIMEOUT_S`); a stuck fit is
  killed and the batch moves on.
- **Auditable** — `batch_logs/manifest_<stamp>.csv` records status + duration per run;
  `batch_logs/batch_<stamp>.log` is the master log.

Edit the grid, MCMC budget (`WARMUP`, `POSTERIOR`), priors, and `TIMEOUT_S` at the
top of `run_all_experiments.py`. Add `"iwls"` to `SAMPLER_GRID` once that sampler is
finalised.

#### Overnight on a laptop

The run dies if the machine sleeps or the terminal closes. Before leaving it:

- Keep it on AC power.
- Set sleep to _Never_ on AC (`powercfg /change standby-timeout-ac 0` on Windows).
- Set lid-close to _Do nothing_ on AC, or leave the lid open.
- Leave the terminal open (the process is a child of it).

---

## bayesm (R) Replication

The bayesm `.R` script lives next to the corresponding Liesel notebook and uses the
renv-managed library. Point it at the **same** dataset the Liesel run used so the two
samplers are provably comparing on identical data, e.g. `5comp_equal.json`.

```bash
# from the repo root, with renv restored
Rscript hbmnl_mixture_experiments/.../bayesm_script.R
```

The script loads the scenario JSON, reconstructs `lgtdata`, runs
`rhierMnlRwMixture`, discards burn-in, and exports `mu`/`cov`/`pvec`/`Delta`/`beta`
draws for the marginal-density comparison. Its model prior
(`ν = n_params + 3`, `V = ν·I`, `Amu`, Dirichlet `a`) matches the Liesel model prior.

---

## Analysing Results

Reload a saved fit into a notebook and feed it straight into `src/analysis.py`:

```python
import pickle, json, pathlib
run = pathlib.Path("hbmnl_mixture_experiments/1_chain/5_comp/NUTS/results/run")

posterior_samples = pickle.load(open(run / "posterior_raw.pkl", "rb"))  # numpy dict
mcmc_results      = pickle.load(open(run / "mcmc_results.pkl", "rb"))    # Goose object
meta              = json.load(open(run / "meta.json"))

K_MODEL, K_TRUE, P = meta["k_model"], meta["k_true"], meta["n_params"]
```

`src/analysis.py` provides component-mean summaries, covariance recovery,
pvec diagnostics, β recovery, and the marginal-density export. The diagnostics that
overlay ground truth take both `K` (= `K_MODEL`, drives the loops) and `K_true`
(guards truth overlays), so spurious components in overspecified fits are labelled
rather than indexed into the (shorter) truth arrays.

`posterior_raw.pkl` reloads anywhere; `mcmc_results.pkl` requires the same
Liesel/JAX environment since it contains live Goose objects.

---

## Adding a New Scenario

Open `hbmnl_mixture_experiments/experiment_configs.py` and add an entry to
`SCENARIOS`:

```python
"2comp_unequal": {
    **BASE,
    "n_components": 2,
    "custom_pvec":  [0.75, 0.25],
},
```

Then generate it:

```bash
uv run python generate_data.py --scenario 2comp_unequal
```

The batch runner picks up new scenarios automatically (it reads `SCENARIOS`).

---

## References

- Rossi, P. E., Allenby, G. M., & McCulloch, R. (2006). _Bayesian Statistics and Marketing_. Wiley.
- Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). _Bayesian Data Analysis_ (3rd ed.). CRC Press.
- Morris, T. P., White, I. R., & Crowther, M. J. (2019). Using simulation studies to evaluate statistical methods. _Statistics in Medicine_, 38(11), 2074–2102.
