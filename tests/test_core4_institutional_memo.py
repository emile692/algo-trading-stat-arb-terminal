from __future__ import annotations

import json
import re
from pathlib import Path

from utils.core4_institutional_memo import Core4InstitutionalMemoOptions, build_core4_institutional_memo


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_core4_institutional_memo_smoke_generates_balanced_pm_and_explainer_memo(tmp_path: Path) -> None:
    result = build_core4_institutional_memo(
        Core4InstitutionalMemoOptions(
            audit_pack_dir=PROJECT_ROOT / "data" / "experiments" / "core4_audit_pack" / "20260604_222941",
            validation_pack_dir=PROJECT_ROOT / "data" / "experiments" / "core4_validation_pack" / "20260514_105610",
            output_dir=tmp_path / "memo_output",
            smoke=True,
        ),
        project_root=PROJECT_ROOT,
    )

    html_path = result["html_path"]
    manifest_path = result["manifest_path"]
    assert html_path.exists()
    assert manifest_path.exists()

    html = html_path.read_text(encoding="utf-8")
    required_sections = [
        "1. Executive Summary",
        "2. What Core 4 Actually Does",
        "3. Investment Universe And Representative Stocks",
        "4. Pair Trading Primer",
        "5. Z-Score, Spread And Statistical Tests",
        "6. Strategy Construction",
        "7. Country Books And Representative Pairs",
        "8. Country Sleeve Deep Dive",
        "9. Portfolio Construction And Diversification",
        "10. Historical Performance And Country Diagnostics",
        "11. Audit Pack And Implementation Readiness",
        "12. Stress Tests",
        "13. From Paper-Ready To Live-Ready",
        "14. Verdict And Recommendation",
        "15. Appendix",
    ]
    for section in required_sections:
        assert section in html

    assert "pair trading" in html.lower()
    assert "z-score" in html.lower()
    assert "engle-granger" in html.lower()
    assert "adf" in html.lower()
    assert "representative stocks" in html.lower()
    assert "Sweden Entry Parameter Research" not in html
    assert "Germany OOS And Overlay Evidence" not in html
    assert "Generated figure" not in html
    assert "paper-ready with limitations" in html

    main_body = html.split("<h2>15. Appendix</h2>", 1)[0]
    assert 8 <= main_body.count("<figure>") <= 10
    assert len(re.findall(r"<h2>\d+\.", html)) == 15

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["section_count"] == 15
    assert 8 <= manifest["figure_count_in_main_memo"] <= 10
    assert "figure_manifest" in manifest
    assert "kept_in_main_memo" in manifest["figure_manifest"]
    assert "moved_to_appendix" in manifest["figure_manifest"]
    assert "excluded" in manifest["figure_manifest"]
    kept_ids = {item["id"] for item in manifest["figure_manifest"]["kept_in_main_memo"]}
    assert "pair_trading_schematic" in kept_ids
    assert "zscore_schematic" in kept_ids
    assert "universe_coverage" in kept_ids
