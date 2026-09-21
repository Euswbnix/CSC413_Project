#!/usr/bin/env python3
"""Assemble the CfC-vs-LSTM power sweep into one table, one test, one figure.

Seeds 0-9 are the lab-fleet runs RE-EVALUATED on the 5090 (runs_val/, a mirror of the
server's lab/), so every seed is scored on one platform and carries the v2 metrics; seeds
10-34 come from runs_power/.
Only the pre-registered configuration is admitted. Runs are keyed by (arm, seed): if the same
seed appears twice under identical config it is one experiment, not two, and counting it twice
would inflate n -- the first copy found (lab fleet before later directories) is kept and the
duplicate is reported.
"""
import csv, json, math, pathlib
import numpy as np

MATCH = dict(lr=0.001, k=0.1, aug="full", T=16, fixed_dt=False, shuffle_frames=False,
             epochs=30, max_steps=None, per_frame_bug=False, translate=True, photometric=True)
# Options added after the lab-fleet runs are absent from their configs; absent means default.
DEFAULTS = dict(loss="mse", dropout=0.0, weight_decay=1e-4)
# One root per directory, fleet first: a seed found in several places keeps the fleet copy.
# LAB is the lab-fleet tree re-evaluated on the 5090: `lab/` on the server, mirrored locally as
# `runs_val/`. Override with --lab / --power.
LAB, POWER = "runs_val", "runs_power"


def roots():
    return [pathlib.Path(LAB, d) for d in ("runs_fleet", "runs", "runs_aug", "runs_lr")] + [pathlib.Path(POWER)]


def admitted(cfg):
    """The pre-registered configuration, checked on every field that changes training. An
    earlier version matched six fields, which let a 15-epoch learning-rate-sweep run through;
    it only stayed out of the table because the fleet copy of that seed happened to be found
    first."""
    return (all(cfg.get(k) == v for k, v in MATCH.items())
            and all(cfg.get(k, v) == v for k, v in DEFAULTS.items()))
PREDICT0_TEST = 13.40


def jl(p):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def collect():
    rows, dups = {}, []
    for root in roots():
        for f in sorted(root.rglob("final_metrics_test.json")):
            d, cfg = jl(f), jl(f.parent / "config.json")
            if not d or not cfg or "rollout" not in d or d.get("arm") not in ("cfc", "lstm"):
                continue
            if not admitted(cfg):
                continue
            key = (d["arm"], cfg["seed"])
            if key in rows:
                if rows[key]["run_dir"] != str(f.parent):
                    dups.append((key, rows[key]["run_dir"], str(f.parent),
                                 rows[key]["test_macro_mae"], d["rollout"]["macro_mae"]))
                continue
            fm = jl(f.parent / "final_metrics.json") or {}
            r = d["rollout"]
            rows[key] = dict(arm=d["arm"], seed=cfg["seed"], run_dir=str(f.parent),
                             best_epoch=d.get("best_epoch", fm.get("best_epoch")),
                             test_macro_mae=r["macro_mae"], test_macro_skill=r["macro_skill"],
                             test_pearson_r=r.get("pearson_r"), test_global_mae=r.get("global_mae"),
                             test_ccc=r.get("ccc"),
                             test_skill_vs_val_constant=r.get("macro_skill_vs_ref_constant"),
                             val_constant=r.get("ref_constant"),
                             test_mae_val_constant=r.get("macro_mae_ref_constant"),
                             test_mae_oracle_constant=r.get("macro_mae_oracle_constant"),
                             constant=r.get("constant_prediction"),
                             winval_macro_skill=(fm.get("val") or {}).get("macro_skill"))
    return rows, dups


def clean(x):
    return np.array([v for v in x if v is not None and not np.isnan(v)], float)


def welch(a, b):
    """Welch t-test with t-distribution p and CI (not the normal approximation: with n~35 per
    arm the normal version is systematically a little anti-conservative)."""
    from scipy import stats
    a, b = clean(a), clean(b)
    na, nb, va, vb = len(a), len(b), a.var(ddof=1), b.var(ddof=1)
    se = math.sqrt(va / na + vb / nb)
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    tc = stats.t.ppf(0.975, df)
    p = stats.ttest_ind(a, b, equal_var=False).pvalue
    d = (a.mean() - b.mean()) / math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    mw = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
    return dict(mean_a=a.mean(), mean_b=b.mean(), med_a=np.median(a), med_b=np.median(b),
                n_a=na, n_b=nb, diff=a.mean() - b.mean(), lo=a.mean() - b.mean() - tc * se,
                hi=a.mean() - b.mean() + tc * se, df=df, p=p, d=d, mw=mw,
                sd_a=a.std(ddof=1), sd_b=b.std(ddof=1))


def holm(ps):
    """Holm step-down adjusted p-values, same order as the input."""
    o = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(o):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def main():
    from scipy import stats
    rows, dups = collect()
    out = sorted(rows.values(), key=lambda r: (r["arm"], r["seed"]))
    pathlib.Path("results").mkdir(exist_ok=True)
    with open("results/power_sweep_per_seed.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader(); w.writerows(out)

    C = [r for r in out if r["arm"] == "cfc"]
    L = [r for r in out if r["arm"] == "lstm"]
    col = lambda R, k: [r[k] for r in R]
    T = {k: welch(col(C, k), col(L, k)) for k in ("test_macro_mae", "test_ccc", "test_pearson_r")}
    adj = dict(zip(T, holm([T[k]["p"] for k in T])))
    vc, vc_mae = C[0]["val_constant"], C[0]["test_mae_val_constant"]
    oc_mae = C[0]["test_mae_oracle_constant"]
    m = T["test_macro_mae"]

    near = lambda R: sum(1 for r in R if r["test_ccc"] is not None and abs(r["test_ccc"]) < 0.005)
    beat0 = lambda R: sum(1 for r in R if r["test_macro_mae"] < PREDICT0_TEST)
    beatc = lambda R: sum(1 for r in R if r["test_macro_mae"] < vc_mae)
    f0 = stats.fisher_exact([[beat0(C), len(C) - beat0(C)], [beat0(L), len(L) - beat0(L)]]).pvalue
    fc = stats.fisher_exact([[beatc(C), len(C) - beatc(C)], [beatc(L), len(L) - beatc(L)]]).pvalue
    rho, rho_p = stats.spearmanr(clean(col([r for r in C if r["test_pearson_r"] is not None
                                             and not np.isnan(r["test_pearson_r"])], "test_macro_mae")),
                                 clean(col(C, "test_pearson_r")))
    r_imp = welch([0.0 if (v is None or np.isnan(v)) else v for v in col(C, "test_pearson_r")],
                  col(L, "test_pearson_r"))
    top = sorted(C, key=lambda r: -(r["test_ccc"] or 0))
    sens = []
    for k in (2, 5):
        keep = top[k:]
        sens.append((k, welch(col(keep, "test_ccc"), col(L, "test_ccc"))["p"],
                     welch(col(keep, "test_pearson_r"), col(L, "test_pearson_r"))["p"]))
    sk = lambda x, c: 1 - x / c

    lines = [
        "# CfC vs 参数配平 LSTM —— 结果（v2 指标）", "",
        "test split，rollout 协议，预注册配置（lr=1e-3 k=0.1 aug=full T=16 epochs=30），"
        "每个 arm 按 seed 去重，全部在同一平台（5090）上评估。Welch t 检验（t 分布，不是正态近似）。", "",
        "## 结论", "",
        f"**预注册主指标 macro MAE 上没有差异**（CfC − LSTM = {m['diff']:+.2f}°，"
        f"95% CI [{m['lo']:+.2f}, {m['hi']:+.2f}]，p = {m['p']:.2f}）。"
        f"**两个 arm 平均都赢不了在 val 上拟合的常数**（对它的 skill：CfC {sk(m['mean_a'], vc_mae):+.3f}，"
        f"LSTM {sk(m['mean_b'], vc_mae):+.3f}）。"
        "看过结果后才加入的探索性指标（r、CCC）上 CfC 名义上更高，但绝对值接近 0、依赖少数几个 seed，"
        "而且主要反映的是 LSTM 更常输出近似常数，不是 CfC 预测得更好。", "",
        "## 主指标（预注册）", "",
        "| | CfC | LSTM | 差 | 95% CI | p (Welch t) | Mann-Whitney p | Cohen d |",
        "|---|---|---|---|---|---|---|---|",
        f"| macro MAE 均值（度） | {m['mean_a']:.3f} | {m['mean_b']:.3f} | {m['diff']:+.3f} | "
        f"[{m['lo']:+.3f}, {m['hi']:+.3f}] | {m['p']:.3f} | {m['mw']:.3f} | {m['d']:+.2f} |",
        f"| macro MAE 中位数（度） | {m['med_a']:.3f} | {m['med_b']:.3f} | {m['med_a']-m['med_b']:+.3f} | | | | |",
        f"| n | {m['n_a']} | {m['n_b']} | | | | | |", "",
        "对任何固定参照的 skill 都只是 macro MAE 的仿射变换，p 值与上表相同，所以不单列成检验：",
        f"对 predict-0 的 skill CfC {sk(m['mean_a'], PREDICT0_TEST):+.4f} / LSTM {sk(m['mean_b'], PREDICT0_TEST):+.4f}；"
        f"对 val 最优常数的 skill CfC {sk(m['mean_a'], vc_mae):+.4f} / LSTM {sk(m['mean_b'], vc_mae):+.4f}。", "",
        f"常数参照（test macro MAE）：predict-0 {PREDICT0_TEST:.2f}；val 上拟合的最优常数 "
        f"c = {vc:+.2f}° → {vc_mae:.2f}；test 上的 oracle 常数 → {oc_mae:.2f}"
        "（看过答案，只作上界，不是对手）。", "",
        "## 探索性指标（看过结果后加入，未预注册）", "",
        "| | CfC 均值 (n) | LSTM 均值 (n) | 中位数 CfC / LSTM | 差 | 95% CI | p | Holm 校正 p |",
        "|---|---|---|---|---|---|---|---|"]
    for k, name in (("test_ccc", "CCC"), ("test_pearson_r", "Pearson r")):
        t = T[k]
        lines.append(f"| {name} | {t['mean_a']:.4f} ({t['n_a']}) | {t['mean_b']:.4f} ({t['n_b']}) | "
                     f"{t['med_a']:.4f} / {t['med_b']:.4f} | {t['diff']:+.4f} | "
                     f"[{t['lo']:+.4f}, {t['hi']:+.4f}] | {t['p']:.4f} | {adj[k]:.4f} |")
    lines += [
        "",
        f"Holm 校正覆盖三个检验（macro MAE、CCC、r）。r 和 CCC 高度相关，基本是同一个问题问了两次。", "",
        "读这两行时必须同时看下面几点：", "",
        f"- **绝对大小可以忽略。** 各 seed r² 的均值：CfC {100*np.nanmean(np.square(clean(col(C,'test_pearson_r')))):.1f}%，"
        f"LSTM {100*np.nanmean(np.square(clean(col(L,'test_pearson_r')))):.1f}%；最好的单个 seed 也只有 "
        f"{100*np.nanmax(np.square(clean(col(C,'test_pearson_r')+col(L,'test_pearson_r')))):.1f}%。"
        "两个模型都几乎没有在跟踪转向信号。",
        f"- **差异主要来自 LSTM 更常输出近似常数。** |CCC| < 0.005 的 seed：CfC {near(C)}/{len(C)}，"
        f"LSTM {near(L)}/{len(L)}。",
        f"- **在 CfC 内部，r 越高 macro MAE 越差**（Spearman ρ = {rho:+.2f}，p = {rho_p:.3f}）。"
        "输出的变化更多，但没有换来更准。",
        f"- **依赖少数几个 seed。** 去掉 CCC 最高的 {sens[0][0]} 个 CfC seed：CCC p = {sens[0][1]:.3f}，"
        f"r p = {sens[0][2]:.3f}；去掉 {sens[1][0]} 个：CCC p = {sens[1][1]:.3f}，r p = {sens[1][2]:.3f}（均未校正）。",
        f"- r 那一行 CfC 是 n={T['test_pearson_r']['n_a']}：塌缩成常数的那个 seed 的 r 没有定义。"
        f"若按 r = 0 计入：差 {r_imp['diff']:+.4f}，p = {r_imp['p']:.4f}。", "",
        "## 计数", "",
        f"- 赢过 predict-0 的 seed：CfC {beat0(C)}/{len(C)}，LSTM {beat0(L)}/{len(L)}（Fisher p = {f0:.3f}，不显著）。",
        f"- 赢过 val 最优常数的 seed：CfC {beatc(C)}/{len(C)}，LSTM {beatc(L)}/{len(L)}（Fisher p = {fc:.2f}）。"
        "两个 arm 都有约 83% 的 seed 比一个常数还差。",
        f"- 严格常数塌缩（输出完全恒定）：CfC {sum(1 for r in C if r['constant'])}/{len(C)}，"
        f"LSTM {sum(1 for r in L if r['constant'])}/{len(L)}；近似常数（|CCC| < 0.005）：CfC {near(C)}/{len(C)}，"
        f"LSTM {near(L)}/{len(L)}。严格标志会低估塌缩。",
        f"- 种子标准差（macro MAE）：CfC {m['sd_a']:.2f}°，LSTM {m['sd_b']:.2f}°。",
        f"- 有 {len(dups)} 个 seed 在多个目录里重复出现，指标逐位相同"
        f"（{sum(1 for *_, a, b in dups if a == b)} 次相同，{sum(1 for *_, a, b in dups if a != b)} 次不同）；"
        "已去重，n 按 seed 计。这些是在不同 git commit 下分别训练的 run，checkpoint 逐字节相同"
        "（同一平台 4080 / torch 2.13），说明训练是确定性的。跨平台则不是：同一批 checkpoint 在 4080 与 5090 上"
        "评估的 test macro MAE 最多相差 0.105°（97 个 run，中位 0.002°），所以本表全部统一在 5090 上评估。"]
    pathlib.Path("results/power_sweep_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    tests = {"test_macro_mae": m, "test_ccc": T["test_ccc"], "test_pearson_r": T["test_pearson_r"]}

    try:
        import matplotlib
    except ImportError:
        # The 5090 host's environment has no matplotlib, and it serves other applications, so
        # nothing is installed into it. The table is the result; the figure is optional.
        print("\nmatplotlib not available: wrote the table and CSV, skipped the figure")
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    rng = np.random.default_rng(0)
    for ax, key, title in ((axes[0], "test_macro_mae", "test macro MAE (deg, lower is better)"),
                           (axes[1], "test_ccc", "test CCC (constant = 0, higher is better)"),
                           (axes[2], "test_pearson_r", "test Pearson r (higher is better)")):
        for i, (R, name, col) in enumerate(((C, "CfC", "#1f77b4"), (L, "LSTM", "#d62728"))):
            v = np.array([r[key] for r in R if r[key] is not None], float)
            v = v[~np.isnan(v)]   # the one collapsed run has r = NaN: undefined, not zero
            x = i + rng.uniform(-0.12, 0.12, len(v))
            ax.scatter(x, v, s=18, alpha=0.6, color=col)
            m = v.mean(); se = v.std(ddof=1) / math.sqrt(len(v))
            ax.errorbar(i + 0.28, m, yerr=1.96 * se, fmt="o", color="black", capsize=4)
        if key == "test_macro_mae":
            ax.axhline(PREDICT0_TEST, ls="--", color="gray", lw=1)
            ax.text(1.55, PREDICT0_TEST, "predict-0", va="bottom", ha="right", fontsize=8, color="gray")
            ax.axhline(vc_mae, ls=":", color="green", lw=1.2)
            ax.text(1.55, vc_mae, f"best constant (val-fit {vc:+.1f})", va="top", ha="right",
                    fontsize=8, color="green")
        else:
            ax.axhline(0, ls="--", color="gray", lw=1)
        t = tests[key]
        tag = "primary" if key == "test_macro_mae" else f"exploratory, Holm p={adj[key]:.3f}"
        ax.set_title(f"{title}\nWelch p={t['p']:.3f}, d={t['d']:+.2f} ({tag})", fontsize=9)
        ax.set_xticks([0, 1]); ax.set_xticklabels([f"CfC (n={t['n_a']})", f"LSTM (n={t['n_b']})"])
        ax.set_xlim(-0.5, 1.6)
    fig.suptitle("CfC vs parameter-matched LSTM, one dot per seed; black = mean with 95% CI", fontsize=10)
    fig.tight_layout()
    pathlib.Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/power_sweep.png", dpi=150)
    print("\nwrote results/power_sweep_per_seed.csv, results/power_sweep_summary.md, figures/power_sweep.png")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab", default=LAB, help="lab-fleet runs re-evaluated on one platform (server: lab)")
    ap.add_argument("--power", default=POWER, help="the seeds 10-34 sweep")
    a = ap.parse_args()
    LAB, POWER = a.lab, a.power
    main()
