"""Figures from verified research outputs; no fitting or allocation changes."""

import json

import numpy as np
import pandas as pd

from . import config as c, evaluation, models


def cumulative_performance(returns):
    """Rebase each scored evaluation period to one without bridging exclusions."""
    selected = returns.loc[returns["scored"]].copy()
    for column in ("execution_close", "realization_close"):
        selected[column] = pd.to_datetime(selected[column], utc=True)
    paths = []
    for (period, name), frame in selected.groupby(["evaluation_period", "portfolio"], sort=False):
        frame = frame.sort_values("execution_close")
        values = frame["weekly_total_return"].to_numpy()
        if (not np.isfinite(values).all() or (values < -1).any()
                or not np.array_equal(frame["realization_close"].iloc[:-1].to_numpy(),
                                      frame["execution_close"].iloc[1:].to_numpy())):
            raise ValueError("Cumulative performance requires contiguous finite scored returns.")
        paths.append(pd.DataFrame({
            "evaluation_period": period, "portfolio": name,
            "market_close": [frame["execution_close"].iloc[0], *frame["realization_close"]],
            "wealth": np.r_[1.0, np.cumprod(1 + values)],
        }))
    if not paths:
        raise ValueError("No scored portfolio returns are available.")
    return pd.concat(paths, ignore_index=True)


def allocation_steps(allocations, end):
    """Executed targets apply from each execution close until the next trade."""
    selected = allocations.loc[allocations["portfolio"].eq("strategy")].copy()
    selected["execution_close"] = pd.to_datetime(selected["execution_close"], utc=True)
    weights = selected.pivot(index="execution_close", columns="asset", values="executed_weight")
    weights = weights.reindex(columns=c.ASSETS).sort_index()
    if (weights.empty or not np.isfinite(weights.to_numpy()).all()
            or not np.allclose(weights.sum(axis=1), 1, atol=c.WEIGHT_TOLERANCE, rtol=0)):
        raise ValueError("Strategy allocations lack a complete fully invested execution.")
    end = pd.Timestamp(end)
    if end <= weights.index[-1]:
        raise ValueError("The final holding endpoint must follow the last execution.")
    return weights.index.append(pd.DatetimeIndex([end])), np.vstack([weights, weights.iloc[-1]])


def export_figures():
    """Rebuild the two published PNGs from the sealed CSV tables."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    frozen = json.loads(c.FROZEN_CONFIG_PATH.read_text(encoding="utf-8"))
    if frozen["seal"] != models.fingerprint({k: v for k, v in frozen.items() if k != "seal"}):
        raise ValueError("Frozen configuration integrity check failed.")
    evaluation._verify_test_receipt(frozen)
    returns = pd.read_csv(c.RESULT_PATHS["weekly_returns"])
    allocations = pd.read_csv(c.RESULT_PATHS["weekly_allocations"])
    paths = cumulative_performance(returns)
    cutoff = pd.Timestamp(frozen["locked_test_cutoff"])
    names = {"strategy": "Strategy", "equal_weight": "Equal weight",
             "inverse_volatility": "Inverse volatility", "cash": "BIL cash"}
    colors = {"strategy": "#2166ac", "equal_weight": "#c47f17",
              "inverse_volatility": "#238b75", "cash": "#777777"}
    periods = ("validation", "locked_test")
    style = {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
             "axes.labelcolor": "#333333", "axes.edgecolor": "#aaaaaa",
             "grid.color": "#dddddd", "grid.linewidth": 0.6, "savefig.dpi": 160}
    c.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True,
                                 gridspec_kw={"width_ratios": [2, 1.25]})
        for axis, period, title in zip(axes, periods, ("Validation · 207 weeks", "Locked test · 104 weeks")):
            sample = paths.loc[paths["evaluation_period"].eq(period)]
            for name, label in names.items():
                line = sample.loc[sample["portfolio"].eq(name)]
                axis.plot(line["market_close"], line["wealth"], label=label, color=colors[name],
                          linewidth=2.1 if name == "strategy" else 1.5,
                          linestyle="--" if name == "cash" else "-")
            axis.set_title(title, fontsize=11, pad=12)
            axis.grid(axis="y")
            locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
            axis.xaxis.set_major_locator(locator)
            axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
            axis.set_xlim(min(cutoff, sample["market_close"].min()) if period == "locked_test"
                          else sample["market_close"].min(), sample["market_close"].max())
        axes[0].set_ylabel("Cumulative wealth · period start = 1")
        axes[1].axvline(cutoff, color="#444444", linestyle=":", linewidth=1)
        fig.suptitle("Weekly portfolio performance", fontsize=15, y=0.98)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.92), ncols=4, frameon=False)
        fig.text(0.5, 0.035, "Gross returns. Periods rebased separately; the boundary-crossing validation holding period is excluded.\n"
                 "Locked-test decision cutoff: 19 July 2024. Curves use weekly execution-to-next-execution returns.",
                 ha="center", fontsize=8.5, color="#555555")
        fig.subplots_adjust(top=0.77, bottom=0.19, left=0.08, right=0.98, wspace=0.08)
        fig.savefig(c.FIGURES_DIR / "performance.png", facecolor="white")
        plt.close(fig)

        end = pd.to_datetime(returns["realization_close"], utc=True).max()
        dates, weights = allocation_steps(allocations, end)
        fig, axis = plt.subplots(figsize=(12, 5))
        axis.stackplot(dates, weights.T, labels=c.ASSETS, step="post",
                       colors=["#2166ac", "#40a6a0", "#deac53", "#9970ab", "#8a9844", "#b7b7b7"])
        axis.axvline(cutoff, color="#222222", linestyle="--", linewidth=1.2)
        for left, right, label in ((dates[0], cutoff, "Validation"), (cutoff, end, "Locked test")):
            axis.text(left + (right-left)/2, 1.025, label, ha="center", va="bottom",
                      transform=axis.get_xaxis_transform(), fontsize=10)
        axis.set(xlim=(dates[0], end), ylim=(0, 1), ylabel="Executed strategy weight")
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        locator = mdates.AutoDateLocator(minticks=5, maxticks=8)
        axis.xaxis.set_major_locator(locator)
        axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        axis.set_axisbelow(True)
        axis.grid(axis="y")
        fig.suptitle("Weekly strategy allocations", fontsize=15, y=0.98)
        fig.legend(*axis.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.91),
                   ncols=6, frameon=False)
        fig.text(0.5, 0.035, "Targets apply at actual execution closes; steps extend to the next execution.\n"
                 "Dashed line: first locked-test decision cutoff, 19 July 2024. Each asset is constrained to 5–50% at execution.",
                 ha="center", fontsize=8.5, color="#555555")
        fig.subplots_adjust(top=0.74, bottom=0.19, left=0.08, right=0.98)
        fig.savefig(c.FIGURES_DIR / "allocations.png", facecolor="white")
        plt.close(fig)
    return [c.FIGURES_DIR / name for name in ("performance.png", "allocations.png")]
