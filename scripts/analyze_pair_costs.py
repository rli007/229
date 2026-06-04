#!/usr/bin/env python3
"""Per-pair cost summary for the report.

Reads the viable-only sweep, and for each of the 15 pairs pulls out the
(accuracy, cost) numbers for the best fixed model, the cheapest fixed model,
the best learned router, and the oracle. Then dumps two artifacts:

  - a summary CSV with per-pair gain/cost columns (used by Table 1)
  - a pgfplots-ready coordinate dump in /tmp (used by Figure 1)

The cost-premium percentages here looked misleading in the early drafts of the
report, so the report itself only uses the absolute USD/1k numbers; the
percentage columns are kept around for sanity checking.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_CSV = PROJECT_ROOT / "results" / "class_viable_only_summary.csv"
OUT_CSV = PROJECT_ROOT / "results" / "class_pair_cost_summary.csv"
OUT_PGF = Path("/tmp/pgfplots_router_points.txt")


def main() -> None:
    summary = pd.read_csv(SUMMARY_CSV)

    rows = []
    for (group, models), sub in summary.groupby(["group", "models"], sort=False):
        fixed = sub[sub["policy"].str.startswith("always_")]
        learned = sub[
            ~sub["policy"].isin(["random", "oracle"])
            & ~sub["policy"].str.startswith("always_")
        ]
        oracle_row = sub[sub["policy"] == "oracle"].iloc[0]

        # short variable names just for readability in the loop body
        bf = fixed.loc[fixed["accuracy_mean"].idxmax()]
        cheap = fixed.loc[fixed["avg_cost_mean"].idxmin()]
        bl = learned.loc[learned["accuracy_mean"].idxmax()]

        gain_pp = (bl["accuracy_mean"] - bf["accuracy_mean"]) * 100
        prem_vs_bf = (
            (bl["avg_cost_mean"] - bf["avg_cost_mean"]) / bf["avg_cost_mean"] * 100
            if bf["avg_cost_mean"] > 0
            else 0.0
        )
        prem_vs_cheap = (
            (bl["avg_cost_mean"] - cheap["avg_cost_mean"])
            / cheap["avg_cost_mean"]
            * 100
            if cheap["avg_cost_mean"] > 0
            else 0.0
        )
        rows.append(
            {
                "pair": group,
                "models": models,
                "best_fixed": bf["policy"].replace("always_", ""),
                "fixed_acc": bf["accuracy_mean"],
                "fixed_cost_usd": bf["avg_cost_mean"],
                "cheapest_fixed": cheap["policy"].replace("always_", ""),
                "cheapest_acc": cheap["accuracy_mean"],
                "cheapest_cost_usd": cheap["avg_cost_mean"],
                "best_learned": bl["policy"].replace("_router", ""),
                "router_acc": bl["accuracy_mean"],
                "router_cost_usd": bl["avg_cost_mean"],
                "gain_pp": gain_pp,
                # the next two columns are kept for sanity, but the report
                # only quotes absolute USD/1k figures
                "cost_premium_vs_best_fixed_pct": prem_vs_bf,
                "cost_premium_vs_cheapest_pct": prem_vs_cheap,
                "oracle_acc": oracle_row["accuracy_mean"],
                "oracle_cost_usd": oracle_row["avg_cost_mean"],
            }
        )
    out = pd.DataFrame(rows).sort_values("gain_pp", ascending=False)
    out.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}\n")

    cols_to_print = [
        "pair",
        "best_fixed",
        "fixed_acc",
        "fixed_cost_usd",
        "best_learned",
        "router_acc",
        "router_cost_usd",
        "gain_pp",
    ]
    print(out[cols_to_print].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # build the pgfplots dump: each row is one pair, with three coordinate
    # tuples (fixed, learned, oracle), x-coords scaled to USD * 1e4 so the
    # axis numbers look reasonable
    print("\n--- pgfplots data: cost (USD x 1e4), accuracy ---")
    lines = ["% pgfplots data: cost_x1e4 accuracy label"]
    for _, r in out.iterrows():
        lines.append(
            f"% {r['pair']}: fixed={r['best_fixed']} "
            f"learned={r['best_learned']} gain={r['gain_pp']:+.2f}pp"
        )
        lines.append(
            f"({r['fixed_cost_usd']*1e4:.4f},{r['fixed_acc']*100:.3f}) "
            f"({r['router_cost_usd']*1e4:.4f},{r['router_acc']*100:.3f}) "
            f"({r['oracle_cost_usd']*1e4:.4f},{r['oracle_acc']*100:.3f})"
        )
    OUT_PGF.write_text("\n".join(lines))
    print(f"wrote {OUT_PGF}")


if __name__ == "__main__":
    main()
