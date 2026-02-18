from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "outputs/runs"


def latest_run_dir() -> Path:
    candidates = sorted([p for p in RUNS.glob("*") if p.is_dir()])
    if not candidates:
        raise RuntimeError("No runs found in outputs/runs")
    return candidates[-1]


def main() -> None:
    run = latest_run_dir()
    log_path = run / "run_log.jsonl"
    rows = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        rows.append(json.loads(line))
    accepted = sum(1 for r in rows if r.get("accepted"))
    avg_score = sum(r.get("critic_scores", {}).get("final_score", 0.0) for r in rows) / max(len(rows), 1)
    report = {
        "run_dir": run.as_posix(),
        "num_windows": len(rows),
        "accepted_windows": accepted,
        "acceptance_rate": accepted / max(len(rows), 1),
        "avg_final_score": avg_score,
    }
    out = ROOT / "outputs/reports/latest_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out.as_posix())


if __name__ == "__main__":
    main()
