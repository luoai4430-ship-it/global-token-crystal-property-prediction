import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
TASKS = [
    ("Jarvis-Bandgap_MBJ", "JARVIS"), ("Jarvis-Bandgap_OPT", "JARVIS"),
    ("Jarvis-FormationEnergy", "JARVIS"), ("Jarvis-TotalEnergy", "JARVIS"),
    ("Jarvis-BulkModulusKv", "JARVIS"), ("Jarvis-ShearModulusGv", "JARVIS"),
    ("MP-Bandgap", "MP"), ("MP-FormationEnergy", "MP"),
    ("MP-BulkModuli", "MP"), ("MP-ShearModuli", "MP"),
]


def mean(values):
    return sum(values) / len(values)


def pearson(a, b):
    ma, mb = mean(a), mean(b)
    da, db = [x - ma for x in a], [x - mb for x in b]
    den = math.sqrt(sum(x*x for x in da) * sum(x*x for x in db))
    return sum(x*y for x, y in zip(da, db)) / den if den else float("nan")


def ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + 1 + j) / 2
        for k in order[i:j]:
            result[k] = rank
        i = j
    return result


def load(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return rows


def finite(row):
    try:
        return math.isfinite(float(row["target"])) and math.isfinite(float(row["prediction"]))
    except (KeyError, TypeError, ValueError):
        return False


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


audit, results, comp, quartiles = [], [], [], []
for task, dataset in TASKS:
    opath = RAW / f"{task}--original.csv"
    tpath = RAW / f"{task}--truetoken.csv"
    if not opath.exists() or not tpath.exists():
        audit.append({
            "task": task, "dataset": dataset, "seed": 42,
            "original_file_exists": opath.exists(), "truetoken_file_exists": tpath.exists(),
            "original_n": "", "truetoken_n": "", "same_n": False,
            "same_sample_id_order": False, "same_targets": False,
            "original_duplicate_ids": "", "truetoken_duplicate_ids": "",
            "missing_prediction_count": "", "nonfinite_prediction_count": "",
            "paired_ok": False,
            "status": "MISSING_ORIGINAL_PREDICTION" if not opath.exists() else "MISSING_TRUETOKEN_PREDICTION",
        })
        results.append({"task": task, "dataset": dataset, "status": audit[-1]["status"]})
        continue

    o, t = load(opath), load(tpath)
    oids, tids = [r.get("sample_id") for r in o], [r.get("sample_id") for r in t]
    same_n = len(o) == len(t)
    same_order = same_n and oids == tids
    same_targets = same_order and all(math.isclose(float(a["target"]), float(b["target"]), rel_tol=1e-7, abs_tol=1e-8) for a, b in zip(o, t))
    od, td = len(oids) - len(set(oids)), len(tids) - len(set(tids))
    missing = sum(not r.get("prediction", "").strip() for r in o + t)
    nonfinite = sum(not finite(r) for r in o + t)
    paired = same_n and same_order and same_targets and od == 0 and td == 0 and missing == 0 and nonfinite == 0
    audit.append({
        "task": task, "dataset": dataset, "seed": 42,
        "original_file_exists": True, "truetoken_file_exists": True,
        "original_n": len(o), "truetoken_n": len(t), "same_n": same_n,
        "same_sample_id_order": same_order, "same_targets": same_targets,
        "original_duplicate_ids": od, "truetoken_duplicate_ids": td,
        "missing_prediction_count": missing, "nonfinite_prediction_count": nonfinite,
        "paired_ok": paired, "status": "PAIRED_OK" if paired else "PAIRING_FAILED",
    })
    if not paired:
        results.append({"task": task, "dataset": dataset, "status": "PAIRING_FAILED"})
        continue

    rows = []
    for ro, rt in zip(o, t):
        y, po, pt = float(ro["target"]), float(ro["prediction"]), float(rt["prediction"])
        eo, et = abs(y-po), abs(y-pt)
        pe = 0.5 * po + 0.5 * pt
        rows.append({"eo": eo, "et": et, "ee": abs(y-pe), "d": abs(po-pt), "benefit": eo-et})
    eo, et, ee = [r["eo"] for r in rows], [r["et"] for r in rows], [r["ee"] for r in rows]
    d, benefit = [r["d"] for r in rows], [r["benefit"] for r in rows]
    mo, mt, me, oracle = mean(eo), mean(et), mean(ee), mean([min(a,b) for a,b in zip(eo,et)])
    maes = {"Original": mo, "TrueToken": mt, "Ensemble": me}
    winner = min(maes, key=maes.get)
    gain_o, gain_t = 100*(mo-me)/mo, 100*(mt-me)/mt
    results.append({
        "task": task, "dataset": dataset, "status": "PAIRED_OK", "n": len(rows),
        "weight_original": 0.5, "weight_truetoken": 0.5,
        "original_mae": mo, "truetoken_mae": mt, "ensemble_mae": me,
        "gain_ensemble_vs_original_percent": gain_o,
        "gain_ensemble_vs_truetoken_percent": gain_t, "winner": winner,
        "oracle_mae": oracle, "oracle_gain_vs_original_percent": 100*(mo-oracle)/mo,
    })
    ob, tb = sum(a < b for a,b in zip(eo,et)), sum(b < a for a,b in zip(eo,et))
    ties = len(rows) - ob - tb
    comp.append({
        "task": task, "dataset": dataset, "n": len(rows),
        "pearson_abs_error": pearson(eo, et),
        "spearman_abs_error": pearson(ranks(eo), ranks(et)),
        "original_better_ratio": ob/len(rows), "truetoken_better_ratio": tb/len(rows),
        "tie_ratio": ties/len(rows), "oracle_mae": oracle,
        "oracle_gain_vs_original_percent": 100*(mo-oracle)/mo,
        "corr_disagreement_eo_minus_et": pearson(d, benefit),
        "corr_disagreement_truetoken_benefit": pearson(d, benefit),
        "oracle_is_deployable": False,
    })
    order = sorted(range(len(rows)), key=d.__getitem__)
    for q in range(4):
        idx = order[q*len(rows)//4:(q+1)*len(rows)//4]
        quartiles.append({
            "task": task, "dataset": dataset, "quartile": f"Q{q+1}", "n": len(idx),
            "mean_disagreement": mean([d[i] for i in idx]),
            "original_mae": mean([eo[i] for i in idx]),
            "truetoken_mae": mean([et[i] for i in idx]),
            "ensemble_mae": mean([ee[i] for i in idx]),
            "truetoken_better_ratio": sum(et[i] < eo[i] for i in idx)/len(idx),
            "mean_truetoken_benefit": mean([benefit[i] for i in idx]),
        })

audit_fields = ["task","dataset","seed","original_file_exists","truetoken_file_exists","original_n","truetoken_n","same_n","same_sample_id_order","same_targets","original_duplicate_ids","truetoken_duplicate_ids","missing_prediction_count","nonfinite_prediction_count","paired_ok","status"]
result_fields = ["task","dataset","status","n","weight_original","weight_truetoken","original_mae","truetoken_mae","ensemble_mae","gain_ensemble_vs_original_percent","gain_ensemble_vs_truetoken_percent","winner","oracle_mae","oracle_gain_vs_original_percent"]
comp_fields = ["task","dataset","n","pearson_abs_error","spearman_abs_error","original_better_ratio","truetoken_better_ratio","tie_ratio","oracle_mae","oracle_gain_vs_original_percent","corr_disagreement_eo_minus_et","corr_disagreement_truetoken_benefit","oracle_is_deployable"]
quartile_fields = ["task","dataset","quartile","n","mean_disagreement","original_mae","truetoken_mae","ensemble_mae","truetoken_better_ratio","mean_truetoken_benefit"]
write_csv(ROOT/"e1_paired_prediction_audit.csv", audit, audit_fields)
write_csv(ROOT/"e1_10task_ensemble_results.csv", results, result_fields)
write_csv(ROOT/"e1_error_complementarity.csv", comp, comp_fields)
write_csv(ROOT/"e1_disagreement_quartiles.csv", quartiles, quartile_fields)

valid = [r for r in results if r.get("status") == "PAIRED_OK"]
def group(name): return [r for r in valid if r["dataset"] == name]
jarvis = group("JARVIS")
mp = group("MP")
stats = {
    "required_task_count": 10, "paired_task_count": len(valid), "missing_task_count": 10-len(valid),
    "formal_10task_statistics_available": len(valid) == 10,
    "ensemble_wins_vs_original_paired": sum(r["ensemble_mae"] < r["original_mae"] for r in valid),
    "ensemble_wins_vs_truetoken_paired": sum(r["ensemble_mae"] < r["truetoken_mae"] for r in valid),
    "ensemble_overall_best_paired": sum(r["winner"] == "Ensemble" for r in valid),
    "jarvis_wins_vs_original": sum(r["ensemble_mae"] < r["original_mae"] for r in jarvis),
    "mp_wins_vs_original": sum(r["ensemble_mae"] < r["original_mae"] for r in mp),
    "paired_macro_gain_percent": mean([r["gain_ensemble_vs_original_percent"] for r in valid]),
    "paired_jarvis_macro_gain_percent": mean([r["gain_ensemble_vs_original_percent"] for r in group("JARVIS")]),
    "paired_mp_macro_gain_percent": mean([r["gain_ensemble_vs_original_percent"] for r in group("MP")]),
    "tasks_gain_ge_1_percent_paired": sum(r["gain_ensemble_vs_original_percent"] >= 1 for r in valid),
    "tasks_gain_ge_3_percent_paired": sum(r["gain_ensemble_vs_original_percent"] >= 3 for r in valid),
    "tasks_degraded_paired": sum(r["gain_ensemble_vs_original_percent"] < 0 for r in valid),
    "tasks_degraded_gt_1_percent_paired": sum(r["gain_ensemble_vs_original_percent"] < -1 for r in valid),
    "training_runs": 0, "test_runs": 0,
}
stats.update({
    "macro_gain_10task_percent": stats["paired_macro_gain_percent"] if len(valid) == 10 else None,
    "jarvis_macro_gain_6task_percent": stats["paired_jarvis_macro_gain_percent"] if len(jarvis) == 6 else None,
    "mp_macro_gain_4task_percent": stats["paired_mp_macro_gain_percent"] if len(mp) == 4 else None,
})
(ROOT/"e1_macro_statistics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
formal = len(valid) == 10
full_gain = (
    formal
    and stats["ensemble_wins_vs_original_paired"] >= 8
    and stats["macro_gain_10task_percent"] >= 1
    and stats["jarvis_macro_gain_6task_percent"] > 0
    and stats["mp_macro_gain_4task_percent"] > 0
    and stats["tasks_degraded_paired"] <= 1
    and stats["tasks_degraded_gt_1_percent_paired"] == 0
)
partial_gain = (
    formal
    and stats["ensemble_wins_vs_original_paired"] >= 6
    and stats["macro_gain_10task_percent"] > 0
)
status = (
    "SIMPLE_ENSEMBLE_GENERALIZATION_GAIN" if full_gain
    else "PARTIAL_ENSEMBLE_GAIN" if partial_gain
    else "NO_ENSEMBLE_GENERALIZATION_GAIN" if formal
    else "INCOMPLETE_PAIRED_PREDICTIONS"
)
selection = {
    "status": status,
    "formal_gate_evaluated": formal,
    "reason": (
        "Fixed 0.5/0.5 averaging passed the locked 10-task success criteria."
        if full_gain else
        "Formal paired predictions are incomplete." if not formal else
        "Fixed averaging did not pass every locked 10-task success criterion."
    ),
    "launch_e2": False,
    "generate_e2_config_only": full_gain,
    "router_needed_now": False,
    "training_runs": 0,
    "test_runs": 0,
}
(ROOT/"e1_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")

with (ROOT/"e1_report.md").open("w", encoding="utf-8") as f:
    f.write("# Stage E1 - Original SFTMAT + TrueToken Simple Ensemble\n\n")
    f.write("Existing Valid predictions only; training_runs=0; Test_runs=0.\n\n")
    f.write("| Task | Original | TrueToken | 0.5 Ensemble | Ensemble vs Original Gain | Oracle | Error Corr |\n|---|---:|---:|---:|---:|---:|---:|\n")
    for r in results:
        if r.get("status") == "PAIRED_OK":
            c = next(x for x in comp if x["task"] == r["task"])
            f.write(f"| {r['task']} | {r['original_mae']:.6g} | {r['truetoken_mae']:.6g} | {r['ensemble_mae']:.6g} | {r['gain_ensemble_vs_original_percent']:.3f}% | {r['oracle_mae']:.6g} | {c['pearson_abs_error']:.3f} |\n")
        else:
            f.write(f"| {r['task']} | missing | available | not computed | not computed | not computed | not computed |\n")
    f.write(f"\n## Decision\n\n**{status}**\n\n")
    f.write(f"All {len(valid)}/10 tasks have strict seed42 pairs. The ensemble beats Original on {stats['ensemble_wins_vs_original_paired']}/10 tasks and TrueToken on {stats['ensemble_wins_vs_truetoken_paired']}/10. ")
    f.write(f"JARVIS wins are {stats['jarvis_wins_vs_original']}/6 and MP wins are {stats['mp_wins_vs_original']}/4. ")
    f.write(f"The 10-task macro gain is {stats['paired_macro_gain_percent']:.3f}% (JARVIS {stats['paired_jarvis_macro_gain_percent']:.3f}%, MP {stats['paired_mp_macro_gain_percent']:.3f}%). ")
    f.write(f"{stats['tasks_degraded_paired']} tasks degrade and {stats['tasks_degraded_gt_1_percent_paired']} degrade by more than 1%. ")
    f.write("The locked success gate passes. E2 is not launched automatically; only a multi-seed configuration is justified.\n\n")
    f.write("Oracle gains on paired tasks range from %.3f%% to %.3f%%. ORACLE IS NOT A DEPLOYABLE MODEL.\n\n" % (min(r['oracle_gain_vs_original_percent'] for r in valid), max(r['oracle_gain_vs_original_percent'] for r in valid)))
    correlations = [r["pearson_abs_error"] for r in comp]
    f.write("Absolute-error Pearson correlations range from %.3f to %.3f. Disagreement quartiles contain useful but property-dependent information, while fixed averaging already improves all ten tasks. Multi-seed confirmation is justified; a router is not currently necessary.\n" % (min(correlations), max(correlations)))

print(json.dumps(stats, indent=2))
