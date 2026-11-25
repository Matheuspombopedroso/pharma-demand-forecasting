# 6. CLI, Docker and CI

**Files:** [`cli.py`](../src/pharma_forecast/cli.py), [`Makefile`](../Makefile),
[`Dockerfile`](../Dockerfile), [`docker-compose.yml`](../docker-compose.yml),
[`.github/workflows/ci.yml`](../.github/workflows/ci.yml)

## CLI

`pip install -e .` installs the `pharma-forecast` command. Each pipeline step
is a sub-command, so it can be run and re-run on its own:

| Command | Does |
|---|---|
| `generate-data [--days N --ingredients K]` | write `data/consumption.csv` |
| `train [--params file.json]` | CV + hold-out + fit + MLflow run |
| `tune [--num-samples N --max-concurrent C --no-asha --no-train]` | Ray Tune search, then `train` with the best params |
| `register [--run-id ID]` | register a run's model (→ `@challenger`) |
| `promote [--version V --force]` | `@challenger` → `@champion` if better |
| `rollback [--version V]` | `@champion` → `@previous` |
| `status` | versions, aliases, hold-out WAPE |
| `predict [--alias A --out file.csv]` | next 7/15/30-day forecast per ingredient |

`train`/`tune` write `outputs/last_run_id.txt`, which `register` reads by
default, so the steps chain without copying ids.

The `Makefile` is a thin wrapper (`make tune SAMPLES=24`). On Windows without
`make`, run the same CLI commands directly.

## Docker

```bash
docker compose up --build
```

- `pipeline` service: runs `generate-data → tune → register → promote →
  status → predict` once and exits. `shm_size: 2gb` because Ray's object store
  lives in `/dev/shm`, and Docker's 64 MB default is too small for it.
- `mlflow` service: starts after the pipeline completes successfully and
  serves the MLflow UI on <http://localhost:5000>, reading the same SQLite
  store from the shared `state` volume.
- All mutable state (data, `mlflow.db`, model artifacts, Ray results) lives
  under `/app/state`, a named volume, so re-running `docker compose up`
  registers a new version next to the old ones, and promote/rollback can be
  tried against real history:

```bash
docker compose run --rm pipeline "pharma-forecast status"
docker compose run --rm pipeline "pharma-forecast rollback"
```

> The Docker files were written but **not run** on the development machine
> (Docker is not installed there). The CI workflow exercises the same CLI
> sequence on Ubuntu without Docker.

## CI (GitHub Actions)

On every push:

1. `ruff check` + `ruff format --check`
2. `pytest`: leakage tests, walk-forward split tests, metrics, the
   train→register→promote→rollback→predict flow on a temporary SQLite store,
   and a 2-trial Ray Tune run
3. Smoke pipeline through the real CLI on 260 days × 2 ingredients with a
   2-trial search

Dependencies are pinned in `requirements.txt` to the versions the tests and
benchmarks ran with.
