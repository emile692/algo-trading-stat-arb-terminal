from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_JOURNAL_DIR = PROJECT_ROOT / "data" / "experiments" / "campaign_journal"
CAMPAIGN_JOURNAL_PATH = CAMPAIGN_JOURNAL_DIR / "campaign_results.md"


def _initial_content() -> str:
    return (
        "# Campaign Results\n\n"
        "Journal synthetique des campagnes de recherche.\n"
        "Chaque bloc est mis a jour de facon idempotente par le script de campagne correspondant.\n\n"
    )


def _normalize_lines(lines: Iterable[str]) -> list[str]:
    out: list[str] = []
    for line in lines:
        txt = str(line).rstrip()
        if txt:
            out.append(txt)
    return out


def upsert_campaign_entry(
    *,
    campaign_key: str,
    title: str,
    summary_lines: Iterable[str],
    out_dir: Path,
    notebook_cell_path: Path | None = None,
) -> Path:
    CAMPAIGN_JOURNAL_DIR.mkdir(parents=True, exist_ok=True)

    if CAMPAIGN_JOURNAL_PATH.exists():
        content = CAMPAIGN_JOURNAL_PATH.read_text(encoding="utf-8")
    else:
        content = _initial_content()

    start_marker = f"<!-- campaign:{campaign_key}:start -->"
    end_marker = f"<!-- campaign:{campaign_key}:end -->"
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = _normalize_lines(summary_lines)
    block_lines = [
        start_marker,
        f"## {title}",
        f"- Updated: {now_str}",
        f"- Outputs: `{out_dir.as_posix()}`",
    ]
    if notebook_cell_path is not None:
        block_lines.append(f"- Notebook cell: `{notebook_cell_path.as_posix()}`")
    block_lines.extend(f"- {line}" for line in lines)
    block_lines.append(end_marker)
    block = "\n".join(block_lines)

    if start_marker in content and end_marker in content:
        start_idx = content.index(start_marker)
        end_idx = content.index(end_marker) + len(end_marker)
        updated = content[:start_idx].rstrip() + "\n\n" + block + content[end_idx:]
    else:
        updated = content.rstrip() + "\n\n" + block + "\n"

    updated = updated.rstrip() + "\n"
    CAMPAIGN_JOURNAL_PATH.write_text(updated, encoding="utf-8")
    return CAMPAIGN_JOURNAL_PATH
