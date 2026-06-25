"""
Copy the analysis notebook template into every completed run folder.

Layout convention: artifacts live in <sampler>/<run>/results/ and the notebook
sits one level up at <sampler>/<run>/liesel.ipynb.  This script finds every
results/ directory that contains posterior_raw.pkl and writes the notebook into
its parent (<run>/).  The template configures itself from meta.json at runtime.

Usage
    uv run python distribute_analysis_notebooks.py            # copy where missing
    uv run python distribute_analysis_notebooks.py --force    # overwrite existing
    uv run python distribute_analysis_notebooks.py --dry-run  # list targets only
"""

import argparse
import pathlib
import shutil
import sys


PROJECT_ROOT = next(
    p for p in [pathlib.Path(__file__).resolve(), *pathlib.Path(__file__).resolve().parents]
    if (p / "pyproject.toml").exists()
)
TEMPLATE = PROJECT_ROOT / "analysis_template.ipynb"
EXP_ROOT = PROJECT_ROOT / "hbmnl_mixture_experiments"
NOTEBOOK_NAME = "liesel.ipynb"       # default name placed in each <run>/ folder


def notebook_name_for(run_dir: pathlib.Path) -> str:
    """bayesm runs live under a BAYESM/ folder and get bayesm.ipynb; the same
    self-configuring template is used either way - only the filename differs so
    the two samplers' notebooks sit side by side without clobbering each other."""
    return "bayesm.ipynb" if "BAYESM" in run_dir.parts else NOTEBOOK_NAME


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="Overwrite an existing notebook.")
    ap.add_argument("--dry-run", action="store_true", help="List target folders and exit.")
    ap.add_argument("--name", default=None,
                    help="Force this filename in every run folder (default: liesel.ipynb, "
                         "or bayesm.ipynb under a BAYESM/ folder).")
    args = ap.parse_args()

    if not TEMPLATE.exists():
        sys.exit(f"Template not found: {TEMPLATE}\n"
                 f"Place analysis_template.ipynb at the repo root first.")

    # Every <run>/ folder — posterior_raw.pkl lives in <run>/results/, so go up two levels.
    run_dirs = sorted({p.parent.parent for p in EXP_ROOT.rglob("posterior_raw.pkl")})

    if not run_dirs:
        sys.exit(f"No run folders found under {EXP_ROOT} (no posterior_raw.pkl files).")

    print(f"Template : {TEMPLATE.relative_to(PROJECT_ROOT)}")
    print(f"Run folders found : {len(run_dirs)}\n")

    copied = skipped = 0
    for d in run_dirs:
        dest = d / (args.name or notebook_name_for(d))
        rel = dest.relative_to(PROJECT_ROOT)
        if args.dry_run:
            mark = "exists" if dest.exists() else "new"
            print(f"  [{mark:>6}] {rel}")
            continue
        if dest.exists() and not args.force:
            print(f"  SKIP (exists)  {rel}")
            skipped += 1
            continue
        shutil.copyfile(TEMPLATE, dest)
        print(f"  WROTE          {rel}")
        copied += 1

    if not args.dry_run:
        print(f"\nDone. {copied} written, {skipped} skipped "
              f"(use --force to overwrite skipped).")


if __name__ == "__main__":
    main()