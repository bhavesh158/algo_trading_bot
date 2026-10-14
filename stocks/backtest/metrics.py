"""Performance statistics for replayed trades."""

from __future__ import annotations

from collections import defaultdict


def summarize(trades: list[dict]) -> dict:
    n = len(trades)
    wins = [t for t in trades if t["net"] > 0]
    gross = sum(t["gross"] for t in trades)
    net = sum(t["net"] for t in trades)
    comm = sum(t["commission"] for t in trades)
    gp = sum(t["net"] for t in wins)
    gl = sum(t["net"] for t in trades if t["net"] <= 0)
    reasons: dict[str, int] = defaultdict(int)
    for t in trades:
        reasons[t["reason"]] += 1
    by_strat: dict[str, dict] = defaultdict(lambda: {"n": 0, "net": 0.0, "wins": 0})
    by_regime: dict[str, dict] = defaultdict(lambda: {"n": 0, "net": 0.0, "wins": 0})
    for t in trades:
        for key, bucket in (("strategy", by_strat), ("regime", by_regime)):
            b = bucket[t[key]]
            b["n"] += 1
            b["net"] += t["net"]
            b["wins"] += 1 if t["net"] > 0 else 0
    return {
        "trades": n,
        "win_rate": len(wins) / n if n else 0.0,
        "gross": gross, "net": net, "commission": comm,
        "expectancy_net": net / n if n else 0.0,
        "expectancy_gross": gross / n if n else 0.0,
        "profit_factor": (gp / abs(gl)) if gl else float("inf") if gp else 0.0,
        "avg_win": gp / len(wins) if wins else 0.0,
        "avg_loss": gl / (n - len(wins)) if n - len(wins) else 0.0,
        "target_hits": sum(1 for t in trades if t["reason"] == "target"),
        "exit_mix": dict(reasons),
        "by_strategy": {k: {**v, "win_rate": v["wins"] / v["n"]} for k, v in by_strat.items()},
        "by_regime": {k: {**v, "win_rate": v["wins"] / v["n"]} for k, v in by_regime.items()},
    }


def print_report(s: dict) -> None:
    print(f"\n{'=' * 64}\nPHASE 1 BACKTEST REPORT (fixed ~Rs.30k notional, Rs.25/side, 0.05% slip)")
    print(f"{'=' * 64}")
    print(f"Trades: {s['trades']} | Win rate: {s['win_rate']:.1%} | "
          f"Target hits: {s['target_hits']} ({s['target_hits'] / s['trades']:.1%})" if s["trades"] else "No trades.")
    print(f"Gross P&L: Rs.{s['gross']:,.2f} | Commission: Rs.{s['commission']:,.2f} | "
          f"Net P&L: Rs.{s['net']:,.2f}")
    print(f"Expectancy/trade: gross Rs.{s['expectancy_gross']:+.2f} | net Rs.{s['expectancy_net']:+.2f}")
    print(f"Profit factor: {s['profit_factor']:.2f} | Avg win Rs.{s['avg_win']:+.2f} | "
          f"Avg loss Rs.{s['avg_loss']:+.2f}")
    print(f"Exit mix: {s['exit_mix']}")
    print("\nBy strategy:")
    for k, v in sorted(s["by_strategy"].items()):
        print(f"  {k:24} n={v['n']:>4} win={v['win_rate']:.0%} net=Rs.{v['net']:>+10.2f}")
    print("By entry regime:")
    for k, v in sorted(s["by_regime"].items()):
        print(f"  {k:16} n={v['n']:>4} win={v['win_rate']:.0%} net=Rs.{v['net']:>+10.2f}")
    print(f"{'=' * 64}\n")
