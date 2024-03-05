LATEXMK ?= latexmk
PYTHON ?= python3

.PHONY: all assets clean

all: assets report.pdf

assets:
	$(PYTHON) scripts/generate_assets.py

REPORT_IMAGES := \
	../outputs/dataset_analysis/plots/porto_gps_density.png \
	../outputs/dataset_analysis/plots/temporal_distributions.png

REPORT_FIGURES := \
	figures/public_score_ranking.png \
	figures/ensemble_repeat_gain.png \
	figures/prefix_length_quantiles.png

REPORT_TABLES := \
	tables/dataset_schema.tex \
	tables/dataset_summary.tex \
	tables/model_comparison.tex

report.pdf: main.tex preamble.tex references.bib $(REPORT_IMAGES) $(REPORT_FIGURES) $(REPORT_TABLES) $(wildcard sections/*.tex)
	mkdir -p build
	$(LATEXMK) -C -outdir=build -jobname=report main.tex
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build -jobname=report main.tex
	cp build/report.pdf report.pdf

clean:
	rm -rf build
