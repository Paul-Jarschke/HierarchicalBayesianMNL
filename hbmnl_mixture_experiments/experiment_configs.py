"""
Simulation scenarios for the HBMNL mixture comparison study.

Design:
  - One equal-weight scenario per K in {1, 2, 3, 5}
  - 300 decision-making units x 30 observations, to roughly match the structure of Rossi (2006) margarine example
Reference: Rossi (2006) §5.5
"""

BASE = dict(
    n_units  = 300,
    n_obs    = 30,
    n_alts   = 4,
    n_demos  = 2,
    seed     = 42,
)

SCENARIOS: dict[str, dict] = {

    "1comp": {
        **BASE,
        "n_components": 1,
        "custom_pvec":  [1.0],
        # K=1 degenerates to standard HMNL — serves as sanity check
        # that both samplers agree on the baseline
    },

    "2comp_equal": {
        **BASE,
        "n_components": 2,
        "custom_pvec":  [0.50, 0.50],
    },

    "3comp_equal": {
        **BASE,
        "n_components": 3,
        "custom_pvec":  [1/3, 1/3, 1/3],
    },

    "5comp_equal": {
        **BASE,
        "n_components": 5,
        "custom_pvec":  [0.20, 0.20, 0.20, 0.20, 0.20],
        # ~60 units per component on average — recovery is genuinely hard.
        # Rossi (2006) explicitly warns about this regime. Differences
        # between samplers here reflect robustness, not just performance.
    },
}