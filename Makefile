CONFIG ?= configs/wmed.yaml

.PHONY: install dev test lint format demo pipeline site clean

install:            ## install package with data-download extra
	pip install -e ".[data]"

dev:                ## install with dev tools and git hooks
	pip install -e ".[dev,data]" && pre-commit install

test:
	pytest

lint:
	ruff check . && ruff format --check .

format:
	ruff format . && ruff check --fix .

demo:               ## credential-free end-to-end run on synthetic data
	submeso all -c configs/demo_synthetic.yaml

pipeline:           ## full real-data pipeline: make pipeline CONFIG=configs/wmed.yaml
	for s in download prepare pairs train evaluate reconstruct validate visualize; do \
		submeso $$s -c $(CONFIG) || exit 1; done

site:               ## export model + data for the web app and preview it: make site CONFIG=configs/demo_synthetic.yaml
	submeso export-web -c $(CONFIG) $(if $(filter %wmed.yaml %blacksea.yaml,$(CONFIG)),--source real,)
	sh scripts/build_site.sh && python -m http.server -d site 8000

clean:              ## remove demo outputs (keeps downloaded data)
	rm -rf runs/demo_synthetic data/demo_synthetic
