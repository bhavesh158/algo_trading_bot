"""Thursday iron fly validation runs.

Usage (from repo root):
    venv/bin/python -m stocks.backtest.fno.run_fly
"""

from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s | %(message)s")

SPLIT = date(2026, 8, 14)


def report(name, rows):
    df = pd.DataFrame(rows)
    if df.empty:
        print(f"{name}: no expiries"); return
    df["day"] = pd.to_datetime(df["expiry"]).dt.date
    print(f"\n### {name} ({len(df)} Thursdays)")
    for label, sub in (("ALL", df), ("TRAIN", df[df.day <= SPLIT]), ("TEST", df[df.day > SPLIT])):
        n = len(sub)
        w = (sub.net > 0).mean() if n else 0
        print(f"  {label:5} n={n:>3} win={w:.0%} exp_net=Rs.{sub.net.mean() if n else 0:+8.2f} "
              f"net=Rs.{sub.net.sum():+10.2f} worst=Rs.{sub.net.min() if n else 0:+10.2f} "
              f"avg_cost=Rs.{sub.cost.mean() if n else 0:.0f}")
    print("  worst weeks:", df.nsmallest(3, "net")[["expiry", "net", "day_range", "iv"]].to_dict("records"))


def main() -> int:
    from stocks.backtest import data as D
    from stocks.backtest.fno import fly as F

    nsei = D.fetch_symbol("^NSEI", "15m", "60d")
    vix = D.fetch_symbol("^INDIAVIX", "1d", "1y")
    print(f"NSEI 15m: {len(nsei)} rows | VIX daily: {len(vix)} rows")

    outdir = Path("stocks/backtest/results")
    outdir.mkdir(parents=True, exist_ok=True)

    for name, kw in (("fly_w200", {"wing_width": 200, "iv_mult": 1.0}),
                     ("fly_w200_iv12", {"wing_width": 200, "iv_mult": 1.2}),
                     ("fly_w300", {"wing_width": 300, "iv_mult": 1.0})):
        rows = F.run_fly(nsei, vix, **kw)
        pd.DataFrame(rows).to_csv(outdir / f"fno_{name}.csv", index=False)
        report(name, rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
