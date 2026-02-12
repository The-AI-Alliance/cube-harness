# Check Eval Status

Show the current status of a running Ray evaluation.

## Instructions

1. **Check Ray cluster status** by running `ray status` to see active nodes, CPU usage, and pending tasks.

2. **Find the latest experiment output directory** under `~/agentlab_results/al2/` (most recent by modification time).

3. **Extract reward summary from all trajectory JSONL files** in that directory. For each trajectory:
   - Parse the JSONL file and find the last `EnvironmentOutput` (entries with a `"reward"` field in `output`)
   - Extract: `reward`, `done`, and `info.message`

4. **Display a summary table** with columns: Task, Reward, Status (DONE/RUNNING), Message

5. **Show aggregate stats**:
   - Total trajectories (completed vs in progress)
   - Success rate (reward > 0 among completed)
   - Average reward among completed
   - Number of pending Ray tasks remaining

Use this Python snippet to extract rewards from trajectory JSONL files:

```python
python3 -c "
import json, glob, os

# Find latest experiment dir
base = os.path.expanduser('~/agentlab_results/al2')
exp_dirs = sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))],
                  key=lambda d: os.path.getmtime(os.path.join(base, d)), reverse=True)
if not exp_dirs:
    print('No experiment directories found')
    exit()
exp_dir = os.path.join(base, exp_dirs[0])
trajectory_dir = os.path.join(exp_dir, 'trajectories')
print(f'Experiment: {exp_dirs[0]}')
print()

done = 0
in_progress = 0
total_reward = 0.0
results = []
for jsonl in sorted(glob.glob(f'{trajectory_dir}/*.jsonl')):
    name = os.path.basename(jsonl).replace('.jsonl','')
    final_reward = 0.0
    final_done = False
    final_msg = ''
    with open(jsonl) as f:
        for line in f:
            d = json.loads(line)
            out = d.get('output', {})
            if 'reward' in out:
                final_reward = out['reward']
                final_done = out['done']
                final_msg = out.get('info', {}).get('message', '')
    if final_done:
        done += 1
        total_reward += final_reward
    else:
        in_progress += 1
    results.append((name, final_reward, final_done, final_msg))

print(f'Completed: {done} | In progress: {in_progress} | Total: {done + in_progress}')
if done > 0:
    successes = sum(1 for _,r,d,_ in results if d and r > 0)
    print(f'Success: {successes}/{done} ({successes/done*100:.0f}%) | Avg reward: {total_reward/done:.2f}')
print()
for name, reward, is_done, msg in results:
    status = 'DONE' if is_done else 'RUNNING'
    # Remove common prefix for readability
    short = name
    for prefix in ['workarena.servicenow.', 'miniwob.']:
        short = short.replace(prefix, '')
    print(f'  {short:50s} {reward:.1f}  {status:8s} {msg}')
"
```
