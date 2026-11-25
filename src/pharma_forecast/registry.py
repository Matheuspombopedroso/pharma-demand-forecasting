"""MLflow Model Registry flow: register, promote, roll back.

Aliases instead of stages
-------------------------
MLflow deprecated the fixed ``Staging``/``Production`` stages in favour of
free-form **aliases** that point at a model version. We use three:

* ``challenger`` -- the most recently registered candidate.
* ``champion``   -- the version ``predict`` serves (``models:/<name>@champion``).
* ``previous``   -- the champion that was replaced by the last promotion;
  this is what ``rollback`` restores.

Lifecycle
---------
::

    register  -> new version N gets @challenger
                 (if no champion exists yet, N also becomes @champion)
    promote   -> compare @challenger vs @champion on hold-out WAPE;
                 if better (or --force): @previous = old champion,
                 @champion = N, @challenger removed
    rollback  -> @champion = @previous (or an explicit --version);
                 the version rolled back from is tagged ``rolled_back=true``

Because consumers resolve ``@champion`` at load time, promotion and rollback
are instant metadata changes: no files are copied and nothing is retrained.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from pharma_forecast.config import (
    CHALLENGER_ALIAS,
    CHAMPION_ALIAS,
    PREVIOUS_ALIAS,
    REGISTERED_MODEL_NAME,
)

# Metric logged by train.py that promotion compares (lower is better).
PROMOTION_METRIC = "holdout_xgboost_wape_mean"


@dataclass
class PromotionDecision:
    """Result of a promotion attempt.

    Attributes:
        promoted: Whether ``champion`` now points at the candidate.
        candidate_version: Version that was evaluated.
        previous_champion: Version that was champion before (None if none).
        candidate_metric: Candidate's hold-out WAPE.
        champion_metric: Old champion's hold-out WAPE (None if none).
        reason: Human-readable explanation.
    """

    promoted: bool
    candidate_version: str
    previous_champion: str | None
    candidate_metric: float
    champion_metric: float | None
    reason: str


def _client(tracking_uri: str) -> MlflowClient:
    """Create a client bound to the given tracking/registry URI."""
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)


def _alias_version(client: MlflowClient, name: str, alias: str) -> str | None:
    """Return the version an alias points to, or None if unset."""
    try:
        # str(): MLflow returns versions as str or int depending on the call.
        return str(client.get_model_version_by_alias(name, alias).version)
    except MlflowException:
        return None


def _version_metric(client: MlflowClient, name: str, version: str) -> float:
    """Read the promotion metric from the training run behind a version."""
    run_id = client.get_model_version(name, version).run_id
    return float(client.get_run(run_id).data.metrics[PROMOTION_METRIC])


def register(run_id: str, tracking_uri: str, name: str = REGISTERED_MODEL_NAME) -> str:
    """Register the model logged by ``run_id`` and mark it as challenger.

    Returns:
        The new version number (as a string, MLflow's convention).
    """
    client = _client(tracking_uri)
    # MLflow 3 stores models as "logged models" with their own URI; train.py
    # records it as a run tag. Fall back to the classic runs:/ URI otherwise.
    model_uri = client.get_run(run_id).data.tags.get("model_uri", f"runs:/{run_id}/model")
    mv = mlflow.register_model(model_uri, name)
    metric = _version_metric(client, name, mv.version)
    client.set_model_version_tag(name, mv.version, PROMOTION_METRIC, f"{metric:.6f}")
    client.set_registered_model_alias(name, CHALLENGER_ALIAS, mv.version)
    # Bootstrap: the very first version has nothing to compete with.
    if _alias_version(client, name, CHAMPION_ALIAS) is None:
        client.set_registered_model_alias(name, CHAMPION_ALIAS, mv.version)
        client.delete_registered_model_alias(name, CHALLENGER_ALIAS)
    return str(mv.version)


def promote(
    tracking_uri: str,
    version: str | None = None,
    force: bool = False,
    name: str = REGISTERED_MODEL_NAME,
) -> PromotionDecision:
    """Promote the challenger (or ``version``) to champion if it is better.

    Args:
        tracking_uri: MLflow tracking/registry URI.
        version: Explicit version to promote; defaults to ``@challenger``.
        force: Promote even if the candidate's hold-out WAPE is worse.
        name: Registered model name.

    Returns:
        A :class:`PromotionDecision` describing what happened.
    """
    client = _client(tracking_uri)
    candidate = version or _alias_version(client, name, CHALLENGER_ALIAS)
    if candidate is None:
        raise ValueError("No challenger to promote; run `register` first or pass --version.")
    champion = _alias_version(client, name, CHAMPION_ALIAS)
    cand_metric = _version_metric(client, name, candidate)

    if champion == candidate:
        return PromotionDecision(
            False, candidate, champion, cand_metric, cand_metric, "Already champion."
        )

    champ_metric = _version_metric(client, name, champion) if champion else None
    better = champ_metric is None or cand_metric < champ_metric
    if not better and not force:
        return PromotionDecision(
            False,
            candidate,
            champion,
            cand_metric,
            champ_metric,
            f"Rejected: hold-out WAPE {cand_metric:.4f} is not better than "
            f"champion's {champ_metric:.4f}. Use --force to override.",
        )

    if champion is not None:
        client.set_registered_model_alias(name, PREVIOUS_ALIAS, champion)
    client.set_registered_model_alias(name, CHAMPION_ALIAS, candidate)
    if _alias_version(client, name, CHALLENGER_ALIAS) == candidate:
        client.delete_registered_model_alias(name, CHALLENGER_ALIAS)
    reason = "Promoted (forced)." if not better else "Promoted: lower hold-out WAPE."
    return PromotionDecision(True, candidate, champion, cand_metric, champ_metric, reason)


def rollback(
    tracking_uri: str, version: str | None = None, name: str = REGISTERED_MODEL_NAME
) -> tuple[str, str]:
    """Point ``champion`` back at ``@previous`` (or an explicit version).

    Returns:
        ``(rolled_back_from, now_champion)`` version numbers.
    """
    client = _client(tracking_uri)
    current = _alias_version(client, name, CHAMPION_ALIAS)
    target = version or _alias_version(client, name, PREVIOUS_ALIAS)
    if current is None or target is None:
        raise ValueError("Nothing to roll back to: no @champion or no @previous version.")
    client.set_registered_model_alias(name, CHAMPION_ALIAS, target)
    # Mark the bad version so it is visible in the UI why it lost the alias.
    client.set_model_version_tag(name, current, "rolled_back", "true")
    if _alias_version(client, name, PREVIOUS_ALIAS) == target:
        client.delete_registered_model_alias(name, PREVIOUS_ALIAS)
    return current, target


def status(tracking_uri: str, name: str = REGISTERED_MODEL_NAME) -> list[dict[str, str]]:
    """List every version with its aliases, metric and tags (for the CLI)."""
    client = _client(tracking_uri)
    # search_model_versions() does not fill in aliases, so read the
    # alias -> version map from the registered model and invert it.
    by_version: dict[str, list[str]] = {}
    for alias, v in client.get_registered_model(name).aliases.items():
        by_version.setdefault(str(v), []).append(alias)
    rows = []
    for mv in client.search_model_versions(f"name='{name}'"):
        rows.append(
            {
                "version": str(mv.version),
                "aliases": ",".join(sorted(by_version.get(str(mv.version), []))) or "-",
                PROMOTION_METRIC: mv.tags.get(PROMOTION_METRIC, "-"),
                "rolled_back": mv.tags.get("rolled_back", "false"),
                "run_id": mv.run_id,
            }
        )
    return sorted(rows, key=lambda r: int(r["version"]))
