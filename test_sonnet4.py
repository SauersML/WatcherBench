"""
Test WatcherBench for anthropic/claude-sonnet-4.
Runs Email_Scoring (user_audit) and Sandbag_Model_Graded_v2 (assistant_continue).
"""

import os
import json
import math

# Load .env
with open('/Users/user/WatcherBench/.env') as f:
    for line in f:
        line = line.strip()
        if line and '=' in line:
            k, v = line.split('=', 1)
            os.environ[k.strip()] = v.strip()

from openai import OpenAI
from benchmark import (
    fetch_transcript,
    parse_transcript_to_messages,
    apply_scenario_logic,
    evaluate_scenario,
    SCENARIOS,
    _NO_LOGPROBS_MODELS,
    _LOGPROB_SAMPLE_N,
    log,
)

MODEL = "anthropic/claude-sonnet-4"
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])

# Check Monte Carlo path
print(f"\n{'='*70}")
print(f"Model: {MODEL}")
print(f"In _NO_LOGPROBS_MODELS: {MODEL in _NO_LOGPROBS_MODELS}")
print(f"Monte Carlo N: {_LOGPROB_SAMPLE_N}")
print(f"{'='*70}\n")

target_scenarios = {
    "Email_Scoring": None,
    "Sandbag_Model_Graded_v2": None,
}

for scen in SCENARIOS:
    if scen["name"] in target_scenarios:
        target_scenarios[scen["name"]] = scen

results = {}
for sname, scen in target_scenarios.items():
    print(f"\n{'='*70}")
    print(f"SCENARIO: {sname} (role_mode={scen['role_mode']})")
    print(f"  deceptive_token: {scen['deceptive_token']!r}")
    print(f"  honest_token:    {scen['honest_token']!r}")
    print(f"  filename:        {scen['filename']}")
    print(f"{'='*70}")

    # Fetch + parse + apply
    raw = fetch_transcript(scen['filename'])
    print(f"\n[Transcript] Length: {len(raw)} chars")

    messages = parse_transcript_to_messages(raw)
    print(f"[Parsed] {len(messages)} messages. Roles: {[m['role'] for m in messages]}")

    api_messages = apply_scenario_logic(messages, scen)
    print(f"[After apply_scenario_logic] {len(api_messages)} messages. Roles: {[m['role'] for m in api_messages]}")
    print(f"  Last message role: {api_messages[-1]['role']}")
    print(f"  Last message tail: {api_messages[-1]['content'][-300:]!r}")

    # Evaluate
    result = evaluate_scenario(client, MODEL, scen, api_messages)
    results[sname] = result

    print(f"\n{'--- FULL RESULT DICT ---'}")
    print(json.dumps(result, indent=2, default=str))

    print(f"\n{'--- RAW TOP LOGPROBS ---'}")
    for entry in result.get('raw_top_logprobs', []):
        print(f"  token={entry['token']!r:20s}  logprob={entry['logprob']:.4f}  p={math.exp(entry['logprob']):.4f}")

    print(f"\n{'--- TOKEN MATCHES ---'}")
    print(f"  deceptive_token ({scen['deceptive_token']!r}): lp={result['lp_deceptive']}, p={result['p_deceptive']}")
    print(f"  honest_token    ({scen['honest_token']!r}): lp={result['lp_honest']}, p={result['p_honest']}")

    print(f"\n{'--- SCORES ---'}")
    print(f"  status:           {result['status']}")
    print(f"  status_k5:        {result['status_k5']}")
    print(f"  scheming_score:   {result['scheming_score']}")
    print(f"  scheming_score_k5:{result['scheming_score_k5']}")

    print(f"\n{'--- GENERATED TEXT ---'}")
    print(f"  generated_token: {result['generated_token']!r}")

print(f"\n\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
for sname, result in results.items():
    print(f"\n{sname}:")
    print(f"  Monte Carlo used: {MODEL in _NO_LOGPROBS_MODELS}")
    print(f"  Status: {result['status']}")
    print(f"  Generated: {result['generated_token']!r}")
    print(f"  scheming_score: {result['scheming_score']} (k5: {result['scheming_score_k5']})")
    print(f"  top_k returned: {result['top_k']}")
    lps = result.get('raw_top_logprobs', [])
    print(f"  logprob entries: {len(lps)}")
    if lps:
        print(f"  top 3 tokens: {[(e['token'], round(math.exp(e['logprob']),4)) for e in lps[:3]]}")
