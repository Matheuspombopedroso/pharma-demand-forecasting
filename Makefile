# Thin wrapper around the `pharma-forecast` CLI (the CLI is the real interface;
# on Windows without `make`, run the same commands directly).
PF ?= pharma-forecast
SAMPLES ?= 20
CONCURRENCY ?= 4

.PHONY: install lint test generate-data train tune register promote rollback status predict pipeline smoke mlflow-ui clean

install:            ## editable install with dev tools
	pip install -r requirements.txt -e ".[dev]"

lint:               ## static checks (same as CI)
	ruff check . && ruff format --check .

test:               ## unit + smoke tests
	pytest

generate-data:      ## write data/consumption.csv (730 days x 8 ingredients)
	$(PF) generate-data

train:              ## train with default params, log to MLflow
	$(PF) train

tune:               ## Ray Tune search, then train with the best params
	$(PF) tune --num-samples $(SAMPLES) --max-concurrent $(CONCURRENCY)

register:           ## register the last trained run (becomes @challenger)
	$(PF) register

promote:            ## @challenger -> @champion if hold-out WAPE is lower
	$(PF) promote

rollback:           ## @champion -> @previous
	$(PF) rollback

status:             ## list registry versions and aliases
	$(PF) status

predict:            ## forecast next 7/15/30 days with @champion
	$(PF) predict --out outputs/forecast.csv

pipeline: generate-data tune register promote predict  ## full run

smoke:              ## tiny end-to-end run (used by CI)
	$(PF) generate-data --days 260 --ingredients 2
	$(PF) tune --num-samples 2 --max-concurrent 2 --n-folds 3 --test-days 14
	$(PF) register
	$(PF) predict

mlflow-ui:          ## browse runs and the registry at http://localhost:5000
	mlflow ui --backend-store-uri sqlite:///mlflow.db

clean:
	rm -rf data outputs mlflow.db mlartifacts ray_results
