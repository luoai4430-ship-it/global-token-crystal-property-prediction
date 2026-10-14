import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
TASKS = ["Jarvis-Bandgap_MBJ", "Jarvis-BulkModulusKv", "MP-Bandgap", "MP-BulkModuli"]


def mean(xs):
    return sum(xs) / len(xs)


def corr(xs, ys):
    mx, my = mean(xs), mean(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    den = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    return sum(x * y for x, y in zip(dx, dy)) / den if den else float("nan")


def load(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return {r["sample_id"]: r for r in csv.DictReader(f)}


summary = []
samples = []
bins = []
for task in TASKS:
    original = load(RAW / f"{task}__original.csv")
    alternative = load(RAW / f"{task}__truetoken.csv")
    if original.keys() != alternative.keys():
        raise RuntimeError(f"sample_id mismatch for {task}")

    rows = []
    for sid in original:
        o, a = original[sid], alternative[sid]
        target_o, target_a = float(o["target"]), float(a["target"])
        if not math.isclose(target_o, target_a, rel_tol=1e-7, abs_tol=1e-8):
            raise RuntimeError(f"target mismatch for {task}/{sid}")
        po, pa = float(o["prediction"]), float(a["prediction"])
        eo, ea = po - target_o, pa - target_o
        ao, aa = abs(eo), abs(ea)
        advantage = ao - aa
        winner = "TrueToken" if advantage > 1e-12 else "Original" if advantage < -1e-12 else "Tie"
        row = {
            "task": task, "sample_id": sid, "target": target_o,
            "original_prediction": po, "truetoken_prediction": pa,
            "original_abs_error": ao, "truetoken_abs_error": aa,
            "signed_error_original": eo, "signed_error_truetoken": ea,
            "prediction_disagreement": abs(po - pa),
            "truetoken_advantage": advantage, "winner": winner,
        }
        rows.append(row)
        samples.append(row)

    n = len(rows)
    original_errors = [r["original_abs_error"] for r in rows]
    alternative_errors = [r["truetoken_abs_error"] for r in rows]
    signed_o = [r["signed_error_original"] for r in rows]
    signed_a = [r["signed_error_truetoken"] for r in rows]
    disagreement = [r["prediction_disagreement"] for r in rows]
    advantage = [r["truetoken_advantage"] for r in rows]
    avg_errors = [abs((r["original_prediction"] + r["truetoken_prediction"]) / 2 - r["target"]) for r in rows]
    oracle_errors = [min(r["original_abs_error"], r["truetoken_abs_error"]) for r in rows]
    mae_o, mae_a = mean(original_errors), mean(alternative_errors)
    mae_avg, mae_oracle = mean(avg_errors), mean(oracle_errors)
    summary.append({
        "task": task, "n": n, "original_wins": sum(r["winner"] == "Original" for r in rows),
        "truetoken_wins": sum(r["winner"] == "TrueToken" for r in rows),
        "ties": sum(r["winner"] == "Tie" for r in rows),
        "truetoken_win_rate_percent": 100 * sum(r["winner"] == "TrueToken" for r in rows) / n,
        "original_mae": mae_o, "truetoken_mae": mae_a, "average_ensemble_mae": mae_avg,
        "oracle_ensemble_mae": mae_oracle,
        "average_gain_vs_original_percent": 100 * (mae_o - mae_avg) / mae_o,
        "oracle_gain_vs_original_percent": 100 * (mae_o - mae_oracle) / mae_o,
        "absolute_error_correlation": corr(original_errors, alternative_errors),
        "signed_error_correlation": corr(signed_o, signed_a),
        "disagreement_advantage_correlation": corr(disagreement, advantage),
    })

    ordered = sorted(rows, key=lambda r: r["prediction_disagreement"])
    for q in range(5):
        part = ordered[q * n // 5:(q + 1) * n // 5]
        bins.append({
            "task": task, "disagreement_quintile": q + 1, "n": len(part),
            "mean_disagreement": mean([r["prediction_disagreement"] for r in part]),
            "truetoken_win_rate_percent": 100 * sum(r["winner"] == "TrueToken" for r in part) / len(part),
            "mean_truetoken_advantage": mean([r["truetoken_advantage"] for r in part]),
        })


def write_csv(name, rows):
    with (ROOT / name).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


write_csv("error_complementarity_summary.csv", summary)
write_csv("error_complementarity_samples.csv", samples)
write_csv("disagreement_quintiles.csv", bins)

macro = {key: mean([float(r[key]) for r in summary]) for key in [
    "truetoken_win_rate_percent", "average_gain_vs_original_percent",
    "oracle_gain_vs_original_percent", "absolute_error_correlation",
    "signed_error_correlation", "disagreement_advantage_correlation",
]}
with (ROOT / "error_complementarity_report.md").open("w", encoding="utf-8") as f:
    f.write("# Valid Error Complementarity: Original vs TrueToken\n\n")
    f.write("Seed 42, paired Valid samples only. No training and no Test forward.\n\n")
    f.write("| Task | N | TrueToken wins | Original MAE | TrueToken MAE | Mean ensemble | Oracle | Abs-error corr | Oracle gain |\n")
    f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in summary:
        f.write(f"| {r['task']} | {r['n']} | {r['truetoken_win_rate_percent']:.1f}% | {r['original_mae']:.6g} | {r['truetoken_mae']:.6g} | {r['average_ensemble_mae']:.6g} | {r['oracle_ensemble_mae']:.6g} | {r['absolute_error_correlation']:.3f} | {r['oracle_gain_vs_original_percent']:.1f}% |\n")
    f.write("\n## Macro diagnostics\n\n")
    for key, value in macro.items():
        f.write(f"- {key}: {value:.4f}\n")
    f.write("\nInterpretation should distinguish oracle headroom from deployable routing: oracle uses labels and is only an upper bound.\n")
    f.write("No Test split was loaded or evaluated.\n")

print((ROOT / "error_complementarity_summary.csv").read_text(encoding="utf-8-sig"))
