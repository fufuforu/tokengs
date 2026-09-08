"""Summarize the offline clustering ablation JSON into a table + plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json",
        default=(
            "workspace/clustering_ablation_unit_shaping_img_2000/"
            "clustering_ablation.json"
        ),
    )
    parser.add_argument("--png", default="")
    args = parser.parse_args()

    payload = json.loads(Path(args.json).read_text(encoding="utf-8"))
    variants = payload["variants"]
    rows = []
    for name, entry in variants.items():
        v = entry["variant"]
        rows.append(
            {
                "name": name,
                "kind": v["kind"],
                "linkage": v.get("linkage", ""),
                "eps": v.get("eps"),
                "min_samples": v.get("min_samples"),
                "min_cluster_size": v.get("min_cluster_size"),
                "pos_w": v.get("pos_w", 1.0),
                "ap25": entry["ap25"],
                "ap50": entry["ap50"],
                "ap75": entry["ap75"],
                "ap": entry["ap_mean"],
                "pred": entry["num_pred"],
                "gt": entry["num_gt"],
                "clusters": entry["num_clusters"],
                "recall05": entry["recall_05"],
                "merge": entry["merging_frac"],
                "frag": entry["fragmentation_frac"],
                "mipc": entry["mean_instances_per_cluster"],
                "mcpi": entry["mean_clusters_per_instance"],
            }
        )
    rows.sort(key=lambda r: -r["ap50"])

    print(
        f"{'variant':28s} {'AP25':>6s} {'AP50':>6s} {'AP75':>6s} {'AP':>6s} "
        f"{'pred':>5s} {'gt':>4s} {'clu':>4s} {'rec.5':>6s} "
        f"{'merge':>6s} {'frag':>6s}"
    )
    for r in rows:
        print(
            f"{r['name']:28s} {r['ap25']:6.3f} {r['ap50']:6.3f} "
            f"{r['ap75']:6.3f} {r['ap']:6.3f} {r['pred']:5.0f} "
            f"{r['gt']:4.0f} {r['clusters']:4.0f} {r['recall05']:6.3f} "
            f"{r['merge']:6.2f} {r['frag']:6.2f}"
        )

    by_kind = {}
    for r in rows:
        by_kind.setdefault(r["kind"], []).append(r)
    print("\nBest per family:")
    for kind, items in by_kind.items():
        best = items[0]
        extra = ""
        if kind == "agg":
            extra = f"linkage={best['linkage']} eps={best['eps']} pw={best['pos_w']}"
        elif kind == "dbscan":
            extra = f"eps={best['eps']} ms={best['min_samples']} pw={best['pos_w']}"
        elif kind == "oracle":
            extra = "GT-instance labels"
        print(
            f"  {kind:8s} {best['name']:28s} AP50={best['ap50']:.3f} "
            f"({extra})"
        )

    base = next(r for r in rows if r["name"] == "agg_avg_eps0.5_pw1")
    oracle = next(r for r in rows if r["name"] == "oracle_gt")
    best_all = rows[0]
    print("\nKey answers:")
    print(f"  baseline (agg avg eps=0.5 pw=1):      AP50={base['ap50']:.3f}")
    print(f"  best clustering variant:               {best_all['name']} "
          f"AP50={best_all['ap50']:.3f}")
    print(f"  oracle (GT-instance labels):           AP50={oracle['ap50']:.3f}")
    print(f"  GS-level oracle (prior diag):          AP50=0.789")
    pw0 = [r for r in rows if r["kind"] == "agg" and r["pos_w"] == 0]
    pw1 = [r for r in rows if r["kind"] == "agg" and r["pos_w"] == 1]
    if pw0 and pw1:
        print(
            f"  agg embedding-only best AP50={max(r['ap50'] for r in pw0):.3f} "
            f"vs embedding+pos best AP50={max(r['ap50'] for r in pw1):.3f}"
        )

    out_csv = Path(args.json).with_suffix(".csv")
    import csv

    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out_csv}")

    if args.png:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        # AP50 vs eps for agg average across pos weights
        for pw, color in ((0.0, "tab:blue"), (1.0, "tab:orange"), (2.0, "tab:green")):
            pts = sorted(
                (r["eps"], r["ap50"])
                for r in rows
                if r["kind"] == "agg"
                and r["linkage"] == "average"
                and r["pos_w"] == pw
            )
            if pts:
                xs, ys = zip(*pts)
                axes[0].plot(
                    xs, ys, marker="o", label=f"pw={pw:g}", color=color
                )
        axes[0].axhline(oracle["ap50"], ls="--", color="gray", label="oracle")
        axes[0].set_title("Agglomerative average: AP50 vs eps")
        axes[0].set_xlabel("eps")
        axes[0].set_ylabel("AP50")
        axes[0].legend()
        # DBSCAN AP50 vs eps by min_samples / pos weight
        for (ms, pw), color in (
            ((5, 0.0), "tab:blue"),
            ((20, 0.0), "tab:cyan"),
            ((5, 1.0), "tab:orange"),
            ((20, 1.0), "tab:red"),
        ):
            pts = sorted(
                (r["eps"], r["ap50"])
                for r in rows
                if r["kind"] == "dbscan"
                and r["min_samples"] == ms
                and r["pos_w"] == pw
            )
            if pts:
                xs, ys = zip(*pts)
                axes[1].plot(
                    xs, ys, marker="o", label=f"ms={ms} pw={pw:g}", color=color
                )
        axes[1].axhline(oracle["ap50"], ls="--", color="gray", label="oracle")
        axes[1].set_title("DBSCAN: AP50 vs eps")
        axes[1].set_xlabel("eps")
        axes[1].legend()
        # bar: top variants
        top = rows[:12]
        axes[2].barh(
            [r["name"] for r in top][::-1],
            [r["ap50"] for r in top][::-1],
        )
        axes[2].axvline(base["ap50"], ls=":", color="gray", label="baseline")
        axes[2].set_title("Top-12 variants by AP50")
        axes[2].set_xlabel("AP50")
        axes[2].legend()
        fig.tight_layout()
        fig.savefig(args.png, dpi=150)
        print(f"wrote {args.png}")


if __name__ == "__main__":
    main()
