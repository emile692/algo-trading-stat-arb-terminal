# utils/portfolio_registry.py
from __future__ import annotations


import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

REGISTRY_PATH = Path("data/portfolio_registry.json")


@dataclass
class PairConfig:
    pair_id: str                 # ex: "AAPL-MSFT"
    asset1: str
    asset2: str
    timeframe: str               # ex: "1H" (ou ce que tu utilises)
    params: dict[str, Any]       # z_entry, z_exit, z_window, wf_train, wf_test, fees, z_stop_mult, etc.
    metrics: dict[str, Any]      # Sharpe_real, Robustness, Sharpe_min, etc.
    created_at: str              # ISO timestamp
    notes: str = ""              # optionnel


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)





def load_registry(path: Path) -> list[dict]:
    """
    Charge le registry depuis JSON.
    Retourne toujours une list[dict].
    Gère les cas où le JSON contient une string encodée, ou params/metrics encodés en string.
    """
    if not Path(path).exists():
        return []

    raw = Path(path).read_text(encoding="utf-8").strip()
    if not raw:
        return []

    data = json.loads(raw)

    # Cas 1: fichier contient une string JSON (double encoding)
    if isinstance(data, str):
        data = json.loads(data)

    if not isinstance(data, list):
        raise ValueError(f"Registry JSON must be a list, got {type(data)}")

    out: list[dict] = []
    for item in data:
        # Cas 2: item est string JSON (liste de strings)
        if isinstance(item, str):
            item = json.loads(item)

        if not isinstance(item, dict):
            continue

        # params/metrics parfois encodés en string
        if "params" in item and isinstance(item["params"], str):
            item["params"] = json.loads(item["params"])
        if "metrics" in item and isinstance(item.get("metrics"), str):
            item["metrics"] = json.loads(item["metrics"])

        out.append(item)

    return out


def save_registry(items: list[dict[str, Any]], path: Path = REGISTRY_PATH) -> None:
    _ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)


def upsert_pair_config(cfg: PairConfig, path: Path = REGISTRY_PATH) -> None:
    items = load_registry(path)

    # upsert sur (pair_id, timeframe)
    found = False
    for i, it in enumerate(items):
        if it.get("pair_id") == cfg.pair_id and it.get("timeframe") == cfg.timeframe:
            items[i] = asdict(cfg)
            found = True
            break

    if not found:
        items.append(asdict(cfg))

    save_registry(items, path)


def remove_pair_config(pair_id: str, timeframe: str, path: Path = REGISTRY_PATH) -> None:
    items = load_registry(path)
    items = [it for it in items if not (it.get("pair_id") == pair_id and it.get("timeframe") == timeframe)]
    save_registry(items, path)
