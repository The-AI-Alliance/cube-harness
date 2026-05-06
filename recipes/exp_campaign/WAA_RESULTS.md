# WAA Campaign Results — for NeurIPS spreadsheet

Mirrors rows 9-10 of `CUBE NeurIPS results - Pass rate.csv` (Windows Agent Arena, full, 152 tasks, 1 seed).

**Snapshot timestamp**: 2026-05-06 ~15:46. v1 (Haiku T1) and v3 (Sonnet T1) are still RUNNING — numbers below are intermediate. v2 (Haiku T2) was killed mid-wave-2. v4 (Sonnet T2) hasn't been retried yet.

## ⚡ Copy-paste rows (TSV → paste into spreadsheet rows 9, 10)

Select cell A9 in the spreadsheet → paste both lines below. Tabs preserve column alignment. Reference-value columns left blank (no published WAA refs to compare against).

```tsv
Windows Agent Arena-PyAutoGUI	full	154	1		41.45% (±7.83pp)	56.58% (±7.88pp)	15.79% (±5.80pp)	25.66% (±6.94pp)	35.53% (±7.61pp)								20.5 (±1.4)	13.6 (±0.8)	28.9 (±2.3)	44.5 (±3.2)	9.3 (±0.8)		$0.27	$0.54	$0.06	$0.42	$0.24
Windows Agent Arena-Computer13	full	154	1		32.89% (±7.47pp)	32.24% (±7.43pp)	7.89% (±4.29pp)	23.68% (±6.76pp)	26.97% (±7.06pp)								29.3 (±2.0)	12.8 (±0.9)	35.9 (±2.8)	40.7 (±2.7)	37.3 (±2.9)		$0.28	$0.18	$0.01	$0.13	$0.34
```

Cell layout per row (27 columns, A → AA):

| Col | Value | Notes |
|-----|-------|-------|
| A | CUBE name | `Windows Agent Arena-PyAutoGUI` / `-Computer13` |
| B | Subset | `full` |
| C | #Tasks | `154` (matches spreadsheet — ours is 152, see caveat) |
| D | n seeds | `1` |
| E | (blank) | spacer |
| F-J | Pass rate × {Haiku, Sonnet, nano, mini, GPT-5.4} | `xx.xx% (±x.xxpp)` (95% CI normal-approx) |
| K | (blank) | spacer |
| L-O | Reference × {Haiku, Sonnet, mini, GPT} | left blank — no published WAA reference |
| P | (blank) | spacer |
| Q-U | Avg steps × 5 models | `mean (±SE)` where `SE = std/sqrt(n_records)` |
| V | (blank) | spacer |
| W-AA | Avg $ × 5 models | `$mean` rounded to 2dp |

## Pass rate

| Tool | Haiku 4.5 | Sonnet 4.6 | GPT-5.4 nano | GPT-5.4 mini | GPT-5.4 |
|------|-----------|------------|--------------|--------------|---------|
| PyAutoGUI (T1) | 41.4% (63/152) ⚙ | 56.6% (86/152) ⚙ | 15.8% (24/152) | 25.7% (39/152) | 35.5% (54/152) |
| Computer13 (T2) | 32.9% (50/152) ⚠ | 32.2% (49/152) ⚠ | 7.9% (12/152) | 23.7% (36/152) | 27.0% (41/152) |

⚙ = run still in progress · ⚠ = run incomplete (v2 killed; v4 wave-1 partial, no wave-2 yet)

## Avg steps / trajectory

| Tool | Haiku 4.5 | Sonnet 4.6 | GPT-5.4 nano | GPT-5.4 mini | GPT-5.4 |
|------|-----------|------------|--------------|--------------|---------|
| PyAutoGUI (T1) | 20.5 ± 16.7 | 13.6 ± 10.1 | 28.9 ± 28.8 | 44.5 ± 39.7 | 9.3 ± 9.4 |
| Computer13 (T2) | 29.3 ± 22.5 | 12.8 ± 8.3 | 35.9 ± 33.8 | 40.7 ± 33.2 | 37.3 ± 35.0 |

(n_records column at bottom — only counts tasks with `episode_record.json` written, so partial runs underrepresented)

## Avg $ / trajectory

| Tool | Haiku 4.5 | Sonnet 4.6 | GPT-5.4 nano | GPT-5.4 mini | GPT-5.4 |
|------|-----------|------------|--------------|--------------|---------|
| PyAutoGUI (T1) | $0.27 ± $0.22 | $0.54 ± $0.57 | $0.06 ± $0.09 | $0.42 ± $0.46 | $0.24 ± $0.37 |
| Computer13 (T2) | $0.28 ± $0.31 | $0.18 ± $0.17 | $0.01 ± $0.02 | $0.13 ± $0.11 | $0.34 ± $0.34 |

## Coverage of metrics tables

How many of the 152 tasks have `episode_record.json` written (cost+steps source). Lower = more in-flight or never-launched tasks.

| Tool | Haiku | Sonnet | nano | mini | GPT |
|------|-------|--------|------|------|-----|
| T1 | 147 | 149 | 152 | 152 | 152 |
| T2 | 121 | 84 | 149 | 149 | 149 |

## Run status

| Run | Dir | Status |
|-----|-----|--------|
| Haiku × T1 (v1) | `20260505_002729_waa_axtree_pyautogui_haiku_v2_waa-cube_b8df7ca8` | RUNNING (wave-2 retry) |
| Haiku × T2 (v2) | `20260505_114348_waa_screenshot_13actions_haiku_v2_waa-cube_d9d475ee` | KILLED mid-wave-2 (50 PASS / 64 r0 / 31 FAILED / 7 abandoned) |
| Sonnet × T1 (v3) | `20260505_115939_waa_axtree_pyautogui_sonnet_v2_waa-cube_6c20bfb1` | RUNNING (wave-2 retry) |
| Sonnet × T2 (v4) | `20260505_155236_waa_screenshot_13actions_sonnet_v2_waa-cube_1f4c63eb` | PENDING wave-2 (only 84/152 records) |
| GPT-5.4 nano × T1 | `20260503_115926_waa_axtree_pyautogui_gpt54nano_waa-cube_6c3052fd` | DONE |
| GPT-5.4 nano × T2 | `20260504_020534_waa_screenshot_13actions_gpt54nano_waa-cube_4529aaec` | DONE (149/152 records) |
| GPT-5.4 mini × T1 | `20260502_121934_waa_axtree_pyautogui_gpt54mini_waa-cube_8c80a1f1` | DONE |
| GPT-5.4 mini × T2 | `20260503_041146_waa_screenshot_13actions_gpt54mini_waa-cube_7fe28396` | DONE (149/152 records) |
| GPT-5.4 × T1 | `20260502_195519_waa_axtree_pyautogui_gpt54_waa-cube_678dbc98` | DONE |
| GPT-5.4 × T2 | `20260503_084927_waa_screenshot_13actions_gpt54_waa-cube_dd65263a` | DONE (149/152 records) |

⚠ Some GPT dirs have duplicate sibling timestamps (e.g. nano-T1 has another at `20260503_113358_..._b5346d93`, mini-T1 has `20260502_121321_..._0d290b7a`). Above I picked the more-populated one — confirm before publishing.

## Metric definitions

- **Pass rate**: `passes / 152` where `passes = (status=='COMPLETED' and reward>0)`. Denominator is 152 even if not every task has a record yet.
- **Avg steps/trajectory**: mean of `episode_record.json -> n_agent_steps` across all tasks with a record; std reported.
- **Avg $/trajectory**: mean of `episode_record.json -> usage.total_cost_usd` across all tasks with a record.
- **152 vs 154**: spreadsheet says 154 — `WAABenchmark` loads 152. 2 tasks dropped somewhere (filter or missing manifest entry). Flag before publishing if it matters.

## How to recompute

```python
import json, glob, os, statistics

def stats(d):
    passes = completed = 0
    steps, costs = [], []
    for ep in glob.glob(f"{d}/episodes/*"):
        if ".archived_" in ep or not os.path.isdir(ep): continue
        sj = os.path.join(ep, "status.json")
        if os.path.exists(sj):
            st = json.load(open(sj))
            if st.get("status") == "COMPLETED":
                completed += 1
                if (st.get("reward") or 0) > 0: passes += 1
        rec = os.path.join(ep, "episode_record.json")
        if os.path.exists(rec):
            r = json.load(open(rec))
            u = r.get("usage", {})
            if u.get("total_cost_usd") is not None: costs.append(u["total_cost_usd"])
            if r.get("n_agent_steps") is not None: steps.append(r["n_agent_steps"])
    return {
        "pass_rate": 100 * passes / 152,
        "avg_steps": statistics.mean(steps) if steps else None,
        "std_steps": statistics.stdev(steps) if len(steps) > 1 else 0,
        "avg_cost": statistics.mean(costs) if costs else None,
        "std_cost": statistics.stdev(costs) if len(costs) > 1 else 0,
        "n_records": len(steps),
    }
```
