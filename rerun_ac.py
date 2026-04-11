#!/usr/bin/env python3
"""Re-run only the assistant_continue cells and splice into results.json."""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import benchmark as b
from openai import OpenAI

for line in open(Path(__file__).parent / ".env"):
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.strip().split("=", 1)
        os.environ.setdefault(k, v)

client = OpenAI(base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"])

results_path = Path(__file__).parent / "results.json"
data = json.load(open(results_path))
results = data["results"]

ac_scenarios = [s for s in b.SCENARIOS if s["role_mode"] == "assistant_continue"]
models = sorted({r["model"] for r in results})
print(f"Re-running {len(ac_scenarios)} scenarios × {len(models)} models = {len(ac_scenarios)*len(models)} cells")

# Prepare messages once per scenario
prepared = {}
for scen in ac_scenarios:
    raw = b.fetch_transcript(scen["filename"])
    msgs = b.parse_transcript_to_messages(raw)
    msgs = b.apply_scenario_logic(msgs, scen)
    prepared[scen["name"]] = (scen, msgs)

# Build lookup of existing results for splice
def key(r):
    return (r["model"], r["scenario"])

new_results = []
for r in results:
    if r["role_mode"] != "assistant_continue":
        new_results.append(r)

coverage = []
for model in models:
    for scen_name, (scen, msgs) in prepared.items():
        print(f"  {model} | {scen_name} ...", flush=True)
        try:
            r = b.evaluate_scenario(client, model, scen, msgs)
        except Exception as e:
            print(f"    ERROR: {e}")
            continue
        new_results.append(r)
        coverage.append({
            "model": model, "scenario": scen_name,
            "status": r["status_k5"],
            "status_full": r["status"],
            "score": r["scheming_score_k5"],
            "score_full": r["scheming_score"],
            "pos": r.get("scored_position"),
            "gen": (r.get("generated_token") or "")[:60],
        })
        time.sleep(0.5)

data["results"] = new_results
data["rerun_ac_timestamp"] = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
with open(results_path, "w") as f:
    json.dump(data, f, indent=2)

print("\n=== COVERAGE ===")
for c in coverage:
    print(f"{c['model']:<32} {c['scenario']:<30} {c['status']:<20} full={c['status_full']:<20} score_full={c['score_full']} pos={c['pos']}")
