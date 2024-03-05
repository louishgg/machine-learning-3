# Taxi destination report

This directory contains the maintainable LaTeX source and reproducible assets for the project technical report.

## Build

From this directory:

```sh
python3 -m pip install -r requirements.txt
make
```

Asset generation requires Python 3.11 or newer.

The build first runs `scripts/generate_assets.py`, which re-scores the saved submissions and writes the three PNG figures and three tables consumed by the report. It does not train models, execute notebooks, or require an external W&B export. It then builds `report.pdf` with `latexmk` and keeps temporary LaTeX output under `build/`.

To remove LaTeX auxiliary files:

```sh
make clean
```

The build uses `python3` and `latexmk` from `PATH`. Override either command if necessary:

```sh
make PYTHON=/path/to/python3
```

The generator requires NumPy, pandas, and Matplotlib.

## Structure

- `main.tex` and `preamble.tex`: document entry point and common formatting
- `sections/`: concise reader-facing narrative
- `scripts/generate_assets.py`: score validation and asset generation
- `requirements.txt`: pinned Python dependencies for asset generation
- `sources/`: compact, reconciled model-selection inputs used by the generator
- `figures/`: the three generated PNG figures used by the report
- `tables/`: generated LaTeX tables
- `references.bib`: external scholarly references
- `build/`: ignored LaTeX and Matplotlib auxiliaries
- `report.pdf`: compiled deliverable

Do not hand-edit generated files under `figures/` or `tables/`; update the generator instead. `make clean` removes the complete build cache while preserving the compiled deliverable.
