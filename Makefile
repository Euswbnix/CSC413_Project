.PHONY: test test-all smoke eda params
test:       ; pytest -q -m "not slow"
test-all:   ; pytest -q
smoke:      ; python scripts/smoke_test.py
params:     ; python scripts/param_table.py
eda:        ; python data/eda_sullychen.py --data-root data/raw
status:     ; python scripts/readme_status.py
figures:    ; bash scripts/pull_figures.sh --with-gif
preprocess: ; python data/preprocess.py --data-root data/raw --out data/processed --verify-gif 23000
download:   ; bash data/download.sh
