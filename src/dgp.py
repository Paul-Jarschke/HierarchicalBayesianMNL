import numpy as np
import os
import json
import warnings

import jax.numpy as jnp
from scipy.stats import invwishart


def generate_mixture_simulated_data(
        n_units=2000, n_obs=100, n_alts=4, n_components=2, n_params=None,
        n_demos=2, custom_pvec=None, custom_indicators=None, seed=123):
    """
    Simulate data for a Bayesian Hierarchical Multinomial Logit model
    with a mixture-of-normals heterogeneity distribution.

    Follows Rossi (2006) §5.5 specification:
      - Z is centred; no intercept column                          (§5.5 advice)
      - mu_k  ~ N(0, I / A_MU),  A_MU = 1/16 for standardised X  (§5.5)
      - Sigma_k ~ IW(nu, V),  nu = nvar+3, V = nu*I               (§5.5)
      - Continuous X attributes are standardised globally before
        choice simulation                                          (§5.5 advice)
      - K >= 4 with n_obs < 50 triggers a UserWarning             (§5.5 caution)
    """

    # ------------------------------------------------------------------
    # FIX 4 — warn early for scenarios Rossi flags as hard to recover
    # ------------------------------------------------------------------
    if n_components >= 4 and n_obs < 50:
        warnings.warn(
            f"K={n_components} with n_obs={n_obs}: Rossi (2006) warns that "
            "recovering complex mixture structure is difficult with low per-unit "
            "observation counts. Component drift is a real risk.",
            UserWarning,
            stacklevel=2,
        )

    np.random.seed(seed)

    if n_params is None:
        n_params = n_alts
    if n_params < n_alts - 1:
        raise ValueError(f"n_params ({n_params}) must be at least n_alts - 1")

    n_ascs      = n_alts - 1
    n_continuous = n_params - n_ascs

    # ------------------------------------------------------------------
    # Demographics — centred, no intercept  (Rossi §5.5)
    # ------------------------------------------------------------------
    Z  = np.random.normal(0, 1, size=(n_units, n_demos))
    Z -= Z.mean(axis=0)                    # column-wise centering

    Delta_true = np.random.normal(0, 0.5, size=(n_demos, n_params))

    # Mixture weights
    if custom_pvec is not None:
        true_pvec  = np.array(custom_pvec, dtype=float)
        true_pvec /= true_pvec.sum()
    else:
        raw_p     = np.random.uniform(0.5, 2.0, n_components)
        true_pvec = raw_p / raw_p.sum()

    # ------------------------------------------------------------------
    # FIX 2 — mu_k: explicit prior-consistent draw with A_mu = 1/16
    # FIX 3 — Sigma_k: IW(nu, V) per Rossi §5.5 (replaces diagonal-uniform)
    # ------------------------------------------------------------------
    # Rossi: "We then set A_mu to 1/16 or so rather than 1/100"
    # This keeps component means within ± ~8 (2 SD) for standardised X.
    A_MU      = 1.0 / 16.0                   # precision → SD per mu_k entry = 4
    NU_SIGMA  = n_params + 3                  # Rossi: nu = nvar + 3
    V_SIGMA   = NU_SIGMA * np.eye(n_params)  # Rossi: V  = nu * I

    true_mu_k    = np.zeros((n_components, n_params))
    true_Sigma_k = np.zeros((n_components, n_params, n_params))

    for k in range(n_components):
        # Sigma_k ~ IW(nu, V)  — full positive-definite, not restricted diagonal
        true_Sigma_k[k] = invwishart.rvs(df=NU_SIGMA, scale=V_SIGMA)
        # mu_k ~ N(0, I / A_MU)  — simplified prior-consistent draw
        # (Full form per Eq. 5.5.3 would be N(mu_bar, Sigma_k / A_MU);
        #  the simplified isotropic version avoids extreme variance
        #  amplification while preserving prior scale intent.)
        true_mu_k[k] = np.random.normal(0.0, np.sqrt(1.0 / A_MU), size=n_params)

    # ------------------------------------------------------------------
    # Individual-level betas
    # ------------------------------------------------------------------
    if custom_indicators is not None:
        true_indicators = np.array(custom_indicators)
    else:
        true_indicators = np.random.choice(n_components, size=n_units, p=true_pvec)

    beta_true = np.zeros((n_units, n_params))
    for i in range(n_units):
        k            = true_indicators[i]
        mu_i         = Z[i] @ Delta_true + true_mu_k[k]
        beta_true[i] = np.random.multivariate_normal(mu_i, true_Sigma_k[k])

    # ------------------------------------------------------------------
    # FIX 1 — pre-generate X, standardise continuous cols globally,
    #          then simulate choices on the standardised design matrix
    # ------------------------------------------------------------------
    # Two-pass approach:
    #   Pass 1 — fill X_array (ASC block fixed; continuous block random)
    #   Standardise — per-attribute, across all (obs × alts)
    #   Pass 2 — compute utilities + draw choices

    n_total = n_units * n_obs
    X_array = np.zeros((n_total, n_alts, n_params))

    flat_idx = 0
    for i in range(n_units):
        for t in range(n_obs):
            X_it = np.zeros((n_alts, n_params))
            # ASC block: alt 0 is the reference category (all zeros)
            for a in range(1, n_alts):
                X_it[a, a - 1] = 1.0
            # Continuous block: raw Uniform(1, 5) — standardised below
            if n_continuous > 0:
                X_it[:, n_ascs:] = np.random.uniform(
                    1.0, 5.0, size=(n_alts, n_continuous)
                )
            X_array[flat_idx] = X_it
            flat_idx += 1

    # Per-attribute standardisation across all observations and alternatives
    # Rossi: "standardising the X variables" so prior on mu_k is interpretable
    if n_continuous > 0:
        for c in range(n_continuous):
            col   = X_array[:, :, n_ascs + c]    # shape (n_total, n_alts)
            mu_c  = col.mean()
            std_c = col.std() + 1e-8
            X_array[:, :, n_ascs + c] = (col - mu_c) / std_c

    # Pass 2 — simulate choices using standardised X
    X_list, y_list, unit_idx_list = [], [], []

    flat_idx = 0
    for i in range(n_units):
        for t in range(n_obs):
            X_it  = X_array[flat_idx]
            U_it  = X_it @ beta_true[i]
            exp_U = np.exp(U_it - U_it.max())    # numerically stable softmax
            probs = exp_U / exp_U.sum()
            y_it  = int(np.random.choice(n_alts, p=probs))
            X_list.append(X_it)
            y_list.append(y_it)
            unit_idx_list.append(i)
            flat_idx += 1

    return {
        "X":               jnp.array(X_list),
        "y":               jnp.array(y_list),
        "Z":               jnp.array(Z),
        "unit_idx":        jnp.array(unit_idx_list),
        "n_units":         n_units,
        "n_params":        n_params,
        "n_demos":         n_demos,
        "K":               n_components,
        "n_alts":          n_alts,
        "TRUE_DELTA":      Delta_true,
        "TRUE_BETA":       beta_true,
        "TRUE_PVEC":       true_pvec,
        "TRUE_MU_K":       true_mu_k,
        "TRUE_SIGMA_K":    true_Sigma_k,
        "TRUE_INDICATORS": true_indicators,
        # Carry DGP hyperparameters forward for metric computation / reporting
        "DGP_A_MU":        float(A_MU),        # precision used for mu_k draws
        "DGP_NU_SIGMA":    int(NU_SIGMA),       # IW degrees of freedom for Sigma_k
    }


# ---------------------------------------------------------------------------
# SAVE HELPER
# ---------------------------------------------------------------------------

def save_to_json(data, filename="sim_data.json"):
    """Converts arrays to lists and saves to JSON."""

    def convert_recursive(obj):
        if isinstance(obj, (np.ndarray, jnp.ndarray)):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert_recursive(v) for k, v in obj.items()}
        if isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
            return obj.item()
        return obj

    serializable_data = convert_recursive(data)

    # Guard against bare filenames where dirname() returns ""
    dir_name = os.path.dirname(filename)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    with open(filename, "w") as f:
        json.dump(serializable_data, f, indent=4)
    print(f"Data successfully saved to:\n{os.path.abspath(filename)}")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    N_UNITS      = 500
    N_OBS        = 30
    N_ALTS       = 4
    N_COMPS      = 5
    N_DEMOS      = 2
    SEED         = 101
    CUSTOM_PVEC  = [0.10, 0.15, 0.20, 0.25, 0.30]

    print("Simulating data...")
    sim_data = generate_mixture_simulated_data(
        n_units      = N_UNITS,
        n_obs        = N_OBS,
        n_alts       = N_ALTS,
        n_components = N_COMPS,
        n_demos      = N_DEMOS,
        custom_pvec  = CUSTOM_PVEC,
        seed         = SEED,
    )

    pvec_str = (
        "".join([str(int(p * 100)) for p in CUSTOM_PVEC])
        if CUSTOM_PVEC else "random"
    )
    filename = (
        f"sim_data_U{N_UNITS}_O{N_OBS}_A{N_ALTS}_K{N_COMPS}"
        f"_D{N_DEMOS}_pvec{pvec_str}_seed{SEED}.json"
    )

    script_dir   = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    save_path    = os.path.join(project_root, "data", "simulated", filename)

    save_to_json(sim_data, save_path)